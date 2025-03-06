## tower-rig

A small implementation of a `tower`-like middleware system with Rig.

## How does it work?

- A service is defined as a struct that implements the `call()` function (as defined by the Service trait) to do something.
- A layer is defined as a struct that implements `Layer<S>`. These can be layered on top of either other layers or services.

Layers are combined together using the `Stack` struct which when using the `layer()` function, will apply the layer from an outer layer to an inner layer. This allows us to recursively add more layers.

We then create a `ServiceBuilder<L>` struct that allows ergonomic stacking of layers on top of a service.

### Implementing your own service

The Service trait so far looks like this:

```rust
trait Service {
    async fn call(&mut self, input: String) -> String;
}
```

All we need to do is implement the `call` function and we're basically ready to go. Note that the input and output are currently concrete types for simplicity.

Consider the following struct which is essentially a wrapper for a Rig agent:

```rust
struct AgentService<M: CompletionModel> {
    agent: Agent<M>,
}

impl<M: CompletionModel> AgentService<M> {
    fn new(agent: Agent<M>) -> Self {
        Self { agent }
    }
}
```

This service is intended to be used as a core service and as such as no inner layer. We can `impl Service` for it like so:

```rust
impl<T: CompletionModel> Service for AgentService<T> {
    async fn call(&mut self, input: String) -> String {
        let input = input.to_string();
        let res = self.agent.prompt(input).await.unwrap();

        res
    }
}
```

### Implementing your own layer

Building on the previous section, we'll go into layers.

Layers are essentially services that can also be recursively stacked on top of each other.

Consider the following struct (which is essentially `AgentService` but as a layer that can call the previous prompt and use it in the next prompt):

```rust
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
```

We can implement a service for it like so (note that `S` is simply the inner service):

```rust
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
```

We can then create layers and add them to a `ServiceBuilder` like so (and then call it, which will eventually return a String):
```rust
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
```
