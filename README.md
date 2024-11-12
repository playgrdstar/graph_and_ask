# Graph & Ask

## Description
`Graph & Ask` is a monorepo project that consists of two main components: `kg_frontend` and `kg_backend`. 
- The frontend is a React-based application designed to visualize and interact with knowledge graphs generated from various data sources. 
- The backend is a Django application that provides the necessary APIs to support the frontend functionalities.

## Quick Start
```bash
git clone https://github.com/playgrdstar/kg_monorepo.git
cd kg_monorepo
./setup.sh
./run.sh
```

## Basic Features and How To Use
- Enter ticker or tickers, separated by commas, in the input field. Maximum of 3 tickers allowed. Window is the number of days to fetch news articles for, and limit is the number of news articles to fetch for each ticker for each day.  A maximum of 5 days and 3 news articles per day is allowed to limit the number of requests to the news API.
- Click the "Generate" button in the section above the input fields. The news articles are fetched from the API, processed, and a knowledge graph is generated for each ticker. The knowledge graphs are streamed to the main centre panel, while the articles are displayed in the side panel on the right.
- Click the "Enrich" button. The knowledge graphs are enriched with semantic information and combined. The combined and enriched knowledge graph is displayed in the main centre panel, while the overall summary of the enriched knowledge graph is displayed in the side panel on the right. Note that this step may take some time if there are many nodes in the knowledge graph. After this step, the context nodes section at the bottom will display all the nodes. Hover over these chips to see the node type and summary. Each node will have the following information: i) a network embedding, ii) a text embedding, and iii) a community assignment after this step.
- For the query step, the context information is dependent on the nodes in the knowledge graph. If no nodes are selected, the default is that all nodes will be in the context, and the query will be answered based on the entire knowledge graph. The backend will filter the most relevant top-N nodes based on the highest semantic similarity to the query based on the network or text embeddings; or iii) in the same community. Otherwise, the query will be answered based on the subset of selected nodes and their connected nodes, as well as  top-N nodes based on the highest semantic similarity to the query based on the network or text embeddings; or iii) in the same community. 
- Click the "Query" button. The answer to the query will then be displayed in the side panel on the right, along with the evidence used to generate the answer.

## kg_frontend

### Description
`kg_frontend` is a React-based frontend application designed to allow the user to visualize and interact with knowledge graphs generated from financial news for different tickers. It involves three key steps:
1. Generate knowledge graphs from news articles. Use side panel on right to view news summaries and their associated knowledge graphs as they are fetched, processed, and generated.
2. Enrich with semantic information and combine all knowledge graphs. After enrichment, each node is enriched with the following semantic information: i) a network embedding, ii) a text embedding, and iii) a community assignment. This step may take some time if there are many nodes in the knowledge graph. An overall summary of the enriched knowledge graph is also generated. Use side panel on right to see overall summary. The context nodes section at the bottom will display all the nodes. Hover over these chips to see the node type and summary.
3. Query the combined knowledge graph to retrieve relevant information, and generate answer. Use knowledge graph visualisation to include the nodes to include in initial context. If no nodes are selected, all nodes are used as initial context. The final context comes from the connected nodes (K-hop) and the top-N nodes based on the semantic similarity of the query to the network embeddings, text embeddings, and the communities of the graph. Use side panel on right to see the answer to the query and evidence.

### What is used for kg_frontend
- React
- TypeScript
- Material-UI
- Axios
- Cytoscape.js

## kg_backend

### Description
`kg_backend` is a Django-based backend application that provides RESTful APIs for the frontend. It handles data processing, storage, and retrieval for knowledge graphs. The backend is responsible for managing user requests, processing data, and serving the frontend application.

### Features
- RESTful API for managing knowledge graphs.
- Integration with various data sources for enriching knowledge graphs.
- User authentication and authorization.
- Middleware for handling CORS and other security features.

### What is used for kg_backend
- Django
- Django Ninja 
- fastnode2vec
- leidenalg
- networkx
- igraph
- sklearn
- pydantic

### Key Files

#### kg_backend

Aside from usual Django files, the key files are:
- **kg_api/views.py**: Defines the API endpoints for the application using Django Ninja. It handles requests related to knowledge graph generation, enrichment, and querying.
- **kg_api/kg_utils.py**: Contains utility functions and classes for processing knowledge graphs, including the `KGGenerator`, `KGEnricher`, and `QueryProcessor`. It also defines data models for articles and knowledge graph nodes.
- **kg_api/data_utils.py**: Provides functions for fetching articles and other data sources necessary for generating knowledge graphs.

### Environment Variables
Add your own keys in the `.env` file:
```
EOD_API_KEY = <From https://eodhd.com/financial-apis/>
FMP_API_KEY = <From https://site.financialmodelingprep.com/>
HF_READ_API_KEY = <From https://huggingface.co/settings/tokens>
OPENAI_API_KEY = <From https://platform.openai.com/api-keys>
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contribution
If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.