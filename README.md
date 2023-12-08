# Dialog Graph Processor

This repository presents a detailed overview of the algorithm implemented in a software component for the automatic generation of script-based dialogue skills based on machine and deep learning models. The component consists of modules for constructing a multi-turn dialogue graph and ranking candidates for dialogue responses.

## Algorithm Overview

The algorithm's workflow begins with the construction of a multi-turn dialogue graph. The process involves encoding dialogue utterances using the sentence encoder, two-stage clustering to create graph nodes, and determining the intentions of dialogue participants. This algorithm allows for the creation of a structure that considers both the content of utterances and their context.

After constructing the dialogue graph, the process of ranking candidates for dialogue responses is described. This process involves analyzing the dialogue context and the multi-turn graph to determine the most suitable responses for the next utterance. Candidates are ranked based on their relevance and suitability in the given context.

## Running the Service

### Graph Construction and Response Prediction Service

To build the service, execute the following command:

```bash
docker build -t dialog_graph_processor:1.0 .
```

To test the service, run the following command:

```bash
docker run -ti --rm dialog_graph_processor:1.0 python3 run_test.py
```

This service processes dialogue graphs and predicts the next dialogue response based on the constructed graph.

**Note:** Ensure that you have Docker installed on your system before running the above commands. If not, please refer to the official [Docker installation guide](https://docs.docker.com/get-docker/).
