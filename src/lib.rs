use std::fmt::Display;

use rig::{
    agent::Agent,
    completion::{CompletionModel, Prompt},
    extractor::Extractor,
};
use schemars::JsonSchema;
use serde::Deserialize;

trait Service {
    type Input: Display;
    type Output: Display;

    async fn call(&mut self, input: Self::Input) -> Self::Output;
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

struct Stack<A, B> {
    inner: A,
    outer: B,
}

impl<A, B> Stack<A, B> {
    fn new(inner: A, outer: B) -> Self {
        Self { inner, outer }
    }
}

impl<S, A, B> Layer<S> for Stack<A, B>
where
    A: Layer<S>,
    B: Layer<A::Service>,
{
    type Service = B::Service;

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
{
    type Input = String;
    type Output = T;

    async fn call(&mut self, input: Self::Input) -> Self::Output {
        self.extractor.extract(&input).await.unwrap()
    }
}

struct AgentService<M: CompletionModel> {
    agent: Agent<M>,
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
    type Input = String;
    type Output = String;

    async fn call(&mut self, input: Self::Input) -> Self::Output {
        println!("Before a message!");
        let res = self.inner.call(input).await;
        println!("LLM response: {res}");

        res.to_string()
    }
}

impl<T: CompletionModel> Service for AgentService<T> {
    type Input = String;
    type Output = String;

    async fn call(&mut self, input: Self::Input) -> Self::Output {
        let res = self.agent.prompt(input).await.unwrap();

        res
    }
}

#[cfg(test)]
mod tests {
    use rig::providers::openai;

    use crate::{AgentService, LoggingMiddleware, Service, ServiceBuilder};

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
}
