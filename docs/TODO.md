# ToDos 
## Presentation
- think of feature showcase order
- explain architecture with diagrams

## Submission
- record video
    - demonstrate required and elective implemented features
    - demonstrate at least one real-world use case that benefitted from elective features
- test setup on fresh system


## Architecture
- flexible multi-layer Configuration
- Framework: Textual TUI

- two fine-tuned models
- graph-rag

- conversation summarization (essence extraction mechanism), should be agent-independent, i.e., a function should be defined, which will be called on some hook, i.e., chat switched


## Features
### Legend
- done

- to do

### Mandatory (Required)
- Chat
    - User Input
        - basic user input
        - select tools
        - switch llm model
        - file / folder upload

        - interrupt generation
        - multiline user input
        - message queueing?
        - ingest button doesnt hide on project collapse
        - keyboard shortcuts
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

- Memory / GraphRAG mechanism
    - recall prior information from groups of chats / Project folder

    - graphrag
    - Add a dense conversation summarization mechanism that creates chunk embeddings of the conversation incrementally (don't recompute)

- ToolSet extension
    - persist tools on file system
    - defined via name, description, executable?
    - refresh tool list every time you press the tools button in the input bar.



    - bug where system prompt is unnecessarily rebuilt in graph
    - remove default project generation
    - investigate performance