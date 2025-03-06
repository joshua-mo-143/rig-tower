use std::{fmt::Display, sync::Arc};

use rig::{
    agent::Agent,
    completion::{CompletionModel, Prompt},
    extractor::Extractor,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

trait Service {
    async fn call(&mut self, input: String) -> String;
}

trait Layer<S> {
    type Service: Service;

    fn layer(&self, inner: S) -> Self::Service;
}

pub struct ServiceBuilder<L> {
    layer: L,
}

impl ServiceBuilder<()> {
    pub fn new() -> Self {
        Self { layer: () }
    }
}

impl<L> ServiceBuilder<L> {
    pub fn layer<N>(self, new_layer: N) -> ServiceBuilder<Stack<L, N>> {
        ServiceBuilder {
            layer: Stack::new(self.layer, new_layer),
        }
    }

    /// Builds the final service by applying all the layers
    /// Note that due to usage of `Stack`, this is possible to be recursive
    pub fn build<S>(self, service: S) -> L::Service
    where
        L: Layer<S>,
    {
        self.layer.layer(service)
    }
}

struct Stack<Inner, Outer> {
    inner: Inner,
    outer: Outer,
}

impl<Inner, Outer> Stack<Inner, Outer> {
    fn new(inner: Inner, outer: Outer) -> Self {
        Self { inner, outer }
    }
}

impl<S, Inner, Outer> Layer<S> for Stack<Inner, Outer>
where
    Inner: Layer<S>,
    Outer: Layer<Inner::Service>,
{
    type Service = Outer::Service;

    fn layer(&self, inner: S) -> Self::Service {
        // Here we stack middleware layers
        // The first layer gets called (the inner layer), then the second layer gets stacked on top of it
        // allowing us to do this recursively
        self.outer.layer(self.inner.layer(inner))
    }
}

struct ExtractService<M: CompletionModel, T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync> {
    extractor: Extractor<M, T>,
}

impl<M, T> Service for ExtractService<M, T>
where
    M: CompletionModel,
    T: Display + JsonSchema + for<'a> Deserialize<'a> + Send + Sync,
    T: Serialize,
{
    async fn call(&mut self, input: String) -> String {
        let res = self.extractor.extract(&input).await.unwrap();

        serde_json::to_string_pretty(&res).unwrap()
    }
}

struct AgentLayerService<M: CompletionModel, S> {
    inner: S,
    agent: Arc<Agent<M>>,
}

impl<M: CompletionModel, S: Service> Service for AgentLayerService<M, S> {
    async fn call(&mut self, input: String) -> String {
        let res = self.inner.call(input).await;

        let next = self.agent.prompt(res.as_ref()).await.unwrap();

        next
    }
}

struct AgentLayer<M: CompletionModel> {
    agent: Arc<Agent<M>>,
}

impl<M: CompletionModel> AgentLayer<M> {
    fn new(agent: Agent<M>) -> Self {
        let agent = Arc::new(agent);

        Self { agent }
    }
}

impl<S: Service, M: CompletionModel> Layer<S> for AgentLayer<M> {
    type Service = AgentLayerService<M, S>;

    fn layer(&self, inner: S) -> Self::Service {
        AgentLayerService {
            inner,
            agent: Arc::clone(&self.agent),
        }
    }
}

struct AgentService<M: CompletionModel> {
    agent: Agent<M>,
}

impl<M: CompletionModel> AgentService<M> {
    fn new(agent: Agent<M>) -> Self {
        Self { agent }
    }
}

impl<M: CompletionModel> From<Agent<M>> for AgentService<M> {
    fn from(value: Agent<M>) -> Self {
        Self::new(value)
    }
}

struct LoggingMiddleware;

impl<S: Service> Layer<S> for LoggingMiddleware {
    type Service = LoggingService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        LoggingService { inner }
    }
}

struct LoggingService<S> {
    inner: S,
}

// Base case: Allow `()` as a no-op layer
impl<S: Service> Layer<S> for () {
    type Service = S;

    fn layer(&self, inner: S) -> Self::Service {
        inner
    }
}

impl<S> Service for LoggingService<S>
where
    S: Service,
{
    async fn call(&mut self, input: String) -> String {
        println!("Before a message!");
        let res = self.inner.call(input).await;
        println!("LLM response: {res}");

        res.to_string()
    }
}

impl<T: CompletionModel> Service for AgentService<T> {
    async fn call(&mut self, input: String) -> String {
        let input = input.to_string();
        let res = self.agent.prompt(input).await.unwrap();

        res
    }
}

#[cfg(test)]
mod tests {
    use rig::providers::openai;

    use crate::{AgentLayer, AgentService, LoggingMiddleware, Service, ServiceBuilder};

    #[tokio::test]
    async fn macro_works() {
        let openai_client = openai::Client::from_env();

        let agent = openai_client
            .agent("gpt-4o")
            .preamble("You are a helpful assistant.")
            .build();

        let agent_service = AgentService { agent };

        let mut thing = ServiceBuilder::new()
            .layer(LoggingMiddleware)
            .build(agent_service);

        thing.call("Hello world!".to_string()).await;
    }

    #[tokio::test]
    async fn agent_layering_works() {
        let openai_client = openai::Client::from_env();

        let agent_one = openai_client
            .agent("gpt-4o")
            .preamble("You are a helpful assistant.")
            .build();

        let agent_two = openai_client
            .agent("gpt-4o")
            .preamble("Your job is to guess what the user prompt is in response to. Only give your guess.")
            .build();

        let agent_service = AgentService::new(agent_one);
        let agent_layer = AgentLayer::new(agent_two);

        let mut service = ServiceBuilder::new()
            .layer(agent_layer)
            .build(agent_service);

        println!("{}", service.call("Hello world!".into()).await)
    }
}
