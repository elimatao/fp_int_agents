## Presentation Script
### Prepare
- kleine anfragen website
- load app
- test pdf
- graph image

### Flow
welcome to BundesRAG, an application that enables you to closely experience our government by talking to an agent training on its answers, i.e. the ones given by (show website)!

before open app: might want to set huggingface api key as base models downloaded from there.

In order to do this, open application

you will see this is a standard chat application

Create Project

Select agent. more on agents later.

we have a classic chat here can write messages (arithmetic...)

settings
    select model picker
        show available models: qwen-2.5-3b 4bit quantized, optionally with adapter to sound more like a german bureaucrat

        then we have a reranker model that was also finetuned with lora on the kleine anfragen data (note skipping details, but see README file)

        Note those Mac OS optimized models, won't run on other platform. But the chat app can interact with any openai compatible endpoint. in the default case a local litellm proxy that can be swapped for lets say a ollama server by just swapping the adress in the config.
    
    select coin flip tool
        show call

        further tools can be added: 
        refreshed when clicking on settings

switch chat
    note chat rename

    ask question about other chat to show cross-project summary.
        constructed by
            for simple agent:
                update a conversation summary on every chat leave
                add all summaries to system prompt

            for BundesRAG
                on chat leave a new chunk of the latest messages is embedded and added to the Qdrant Vector database
                at query time, retrieved on-demand

go back to first chat -> show summary

BundesRAG: create project
    RAG based, means agent more complex: show graph. explain

    for rag to be useful, we need data. (there is script for downloading) -> ingest document folder (show document)

    once ingested, we can ask questions.

