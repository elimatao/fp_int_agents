# Features

## Architecture
- flexible multi-layer Configuration
- Framework: Textual TUI

## Features
- Chat
    - User Input
        - basic user input
        - select tools
        - switch llm model
        - file / folder upload
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
- Memory mechanism
    - recall prior information from groups of chats / Project folder
    - Add a dense conversation summarization mechanism that creates chunk embeddings of the conversation incrementally
- ToolSet extension
    - persist tools on file system
    - defined via name, description, executable?
    - refresh tool list every time you press the tools button in the input bar.
    - remove default project generation