# Features

## Submission
- record video
    - demonstrate required and elective implemented features
    - demonstrate at least one real-world use case that benefitted from elective features
- test setup on fresh system


## Architecture
- flexible multi-layer Configuration
- Framework: Textual TUI
- conversation summarization (essence extraction mechanism), should be agent-independent, i.e., a function should be defined, which will be called on some hook, i.e., chat switched

## Presentation Script
welcome to BundesRAG, an application that enables you to closely experience our government by directly talking to it!






## Features
### Legend
- done

- to do

- todo (nice to have)

### Mandatory (Required)
- Chat
    - User Input
        - basic user input
        - select tools
        - switch llm model
        - file / folder upload

        - interrupt generation
        
        - ingest button doesnt hide on project collapse
        - interface responsive as often as possible
    - Conversation History
        - Scrolling
        - Scroll stop on long generations
        - history block visualization / collapsing? e.g. toolcalls, 
        
- Conversation Management
    - switch
    - create
    - continue
    - remove
    
- Project Management
    - create
    - delete project
    - configure

    - showcase agent-binding (for type of conversation )
    - update mutable attributes

- Memory mechanism
    - recall prior information from groups of chats / Project folder

    - Add a dense conversation summarization mechanism that creates chunk embeddings of the conversation incrementally (don't recompute)

- ToolSet extension
    - persist tools on file system
    - defined via name, description, executable?
    - refresh tool list every time you press the tools button in the input bar.
    - remove default project generation


    - bug where system prompt is unnecessarily rebuilt in graph
    - investigate performance
