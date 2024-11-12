import React, { useEffect, useState, useMemo } from "react";
import { Article, KnowledgeGraph, QueryResponse, KGNode } from "./types/api.types";
import { generateKG, enrichKG, queryKG } from "./services/api";
import ArticlePanel from "./components/ArticlePanel";
import GraphVisualization from "./components/GraphVisualization";
import { Box, Stack, Typography, TextField, Button, IconButton, Divider, CircularProgress, Drawer, Chip, Link, Autocomplete } from "@mui/material";
import PanoramaFishEyeIcon from '@mui/icons-material/PanoramaFishEye';
import AdjustIcon from '@mui/icons-material/Adjust';
import KeyboardDoubleArrowLeftIcon from '@mui/icons-material/KeyboardDoubleArrowLeft';
import CloseIcon from '@mui/icons-material/Close';
import LooksOneIcon from '@mui/icons-material/LooksOne';
import LooksTwoIcon from '@mui/icons-material/LooksTwo';
import Looks3Icon from '@mui/icons-material/Looks3';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import AddCircleIcon from '@mui/icons-material/AddCircle';
import RemoveCircleIcon from '@mui/icons-material/RemoveCircle';
import Tooltip from "@mui/material/Tooltip";
import lookup from "./data/lookup.json";

type GenerationProgress = {
    current: number;
    total: number;
} | null;

interface StreamingUpdate {
    nodes: Set<string>;
    timestamp: number;
}

interface Company {
    symbol: string;
    name: string;
    sector: string;
    subSector: string;
    headQuarter: string;
    dateFirstAdded: string | null;
    cik: string;
    founded: string;
}

type CompanyOption = {
    symbol: string;
    name: string;
    sector: string;
};

const MAX_TICKERS = 3;
const MAX_WINDOW = 5;
const MAX_ARTICLES = 3;

const NodeDetails: React.FC<{ node: KGNode }> = ({ node }) => {
    return (
        <Box sx={{ mb: 1 }}>
            {/* <Typography variant="h6" gutterBottom>
                Node Details
            </Typography> */}
            {/* <Typography variant="subtitle2" gutterBottom>
                ID: {node.id}
            </Typography> */}
            <Typography variant="subtitle2" gutterBottom>
                Type: {node.type}
            </Typography>
            <Typography variant="subtitle2" gutterBottom>
                Summary: {node.summary}
            </Typography>
            {node.community !== null && (
                <Typography variant="subtitle2" gutterBottom>
                    Community: {node.community}
                </Typography>
            )}
            {node.articles && node.articles.length > 0 && (
                <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                        Related Articles:
                    </Typography>
                    {node.articles.map((article, index) => (
                        <Link href={article} key={index} underline="hover" variant="body2" gutterBottom>
                            {article}
                        </Link>
                    ))}
                </Box>
            )}
        </Box>
    );
};

// Debounce function
const useDebounce = (value: string, delay: number) => {
    const [debouncedValue, setDebouncedValue] = useState(value);

    useEffect(() => {
        const handler = setTimeout(() => {
            setDebouncedValue(value);
        }, delay);

        return () => {
            clearTimeout(handler);
        };
    }, [value, delay]);

    return debouncedValue;
};

const MemoizedGraphVisualization = React.memo(GraphVisualization, (prevProps, nextProps) => {
    // Only re-render if these props change
    return (
        prevProps.data.updateId === nextProps.data.updateId &&
        prevProps.selectedNodes === nextProps.selectedNodes
    );
});

const App: React.FC = () => {
    const [kgData, setKgData] = useState<KnowledgeGraph | null>(null);
    const [kgId, setKgId] = useState<string | null>(null);
    const [tickers, setTickers] = useState<string>("");
    const [window, setWindow] = useState<number>(1);
    const [limit, setLimit] = useState<number>(1);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [completedSteps, setCompletedSteps] = useState<{
        generate: boolean;
        enrich: boolean;
        query: boolean;
    }>({
        generate: false,
        enrich: false,
        query: false,
    });
    const [query, setQuery] = useState<string>("");
    
    const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
    const [selectedNodes, setSelectedNodes] = useState<Set<string>>(new Set());
    const [queryResponse, setQueryResponse] = useState<QueryResponse | null>(null);
    const [isRightPanelOpen, setIsRightPanelOpen] = useState<boolean>(true);
    const [isLeftPanelOpen, setIsLeftPanelOpen] = useState<boolean>(true);
    const [generationProgress, setGenerationProgress] = useState<GenerationProgress>(null);
    const [eventSource, setEventSource] = useState<EventSource | null>(null);
    const [streamingUpdates, setStreamingUpdates] = useState<StreamingUpdate | null>(null);
    const [kgIds, setKgIds] = useState<string[]>([]);
    

    const [localQuery, setLocalQuery] = useState<string>("");
    const debouncedQuery = useDebounce(localQuery, 500);

    const [topN, setTopN] = useState<number>(5); // State for top N
    const [numHops, setNumHops] = useState<number>(1); // State for number of hops

    const companyOptions = useMemo(() => {
        return lookup.sp500.map(company => ({
            symbol: company.symbol,
            name: company.name,
            sector: company.sector
        }));
    }, []);

    const handleGenerateKG = async () => {
        if (!tickers.trim()) return;
        
        try {
            setIsLoading(true);
            setGenerationProgress(null);
            setKgData(null);  // Reset existing KG data
            
            // Close existing EventSource if any
            if (eventSource) {
                eventSource.close();
                setEventSource(null);
            }

            const params = new URLSearchParams({
                tickers,
                window: window.toString(),
                limit: limit.toString()
            });

            const apiUrl = `http://127.0.0.1:8000/api/generate?${params.toString()}`;
            console.log("[SSE] Connecting to:", apiUrl);

            // Create new EventSource without initial fetch check
            const newEventSource = new EventSource(apiUrl);
            let reconnectAttempts = 0;
            const MAX_RECONNECT_ATTEMPTS = 3;

            newEventSource.onopen = () => {
                console.log("[SSE] Connection opened, readyState:", newEventSource.readyState);
                reconnectAttempts = 0;  // Reset reconnect attempts on successful connection
            };

            newEventSource.onmessage = (event) => {
                console.log("[SSE] Raw message received:", event.data);
                
                try {
                    const data = JSON.parse(event.data);
                    console.log("[SSE] Parsed message:", data);
                    
                    switch (data.type) {
                        case "connection":
                            console.log("[SSE] Connection confirmed");
                            break;

                        case "progress":
                            console.log("[SSE] Progress update:", data);
                            if (data.progress) {
                                setGenerationProgress(data.progress);
                            }
                            break;

                        case "kg_update":
                            console.log("[SSE] Received KG update:", data);
                            if (data.data && Array.isArray(data.data.nodes)) {
                                const currentKgId = data.kg_id;
                                
                                if (currentKgId) {
                                    setKgIds((prevIds) => {
                                        console.log("[App] Updating KG IDs:", {
                                            currentId: currentKgId,
                                            prevIds: prevIds,
                                        });
                                        
                                        if (!prevIds.includes(currentKgId)) {
                                            const newIds = [...prevIds, currentKgId];
                                            console.log("[App] Added new KG ID, updated list:", newIds);
                                            return newIds;
                                        }
                                        return prevIds;
                                    });

                                    setKgId((prevId) => {
                                        console.log("[App] Updating current KG ID:", {
                                            prevId: prevId,
                                            newId: currentKgId
                                        });
                                        return currentKgId;
                                    });
                                }

                                setKgData(prevKG => {
                                    const newKgData = {
                                        nodes: [...(prevKG?.nodes || []), ...data.data.nodes],
                                        edges: [...(prevKG?.edges || []), ...data.data.edges],
                                        articles: [...(prevKG?.articles || []), data.data.article],
                                        summary: prevKG?.summary || "",
                                        updateId: Date.now(),
                                        kg_id: data.kg_id
                                    };
                                    
                                    console.log("[App] Setting new KG state:", {
                                        timestamp: new Date().toISOString(),
                                        prevNodes: prevKG?.nodes.length || 0,
                                        newNodes: newKgData.nodes.length,
                                        updateId: newKgData.updateId,
                                        kgId: data.kg_id  // Log KG ID for debugging
                                    });

                                    return newKgData;
                                });
                            }
                            break;

                        case "complete":
                            console.log("[SSE] Generation complete");
                            if (data.data.kg_id) {
                                setKgId(data.data.kg_id);
                                setKgIds(prev => [...prev, data.data.kg_id]);
                                console.log("[App] Set KG ID:", data.data.kg_id);
                            }
                            setCompletedSteps(prev => ({ ...prev, generate: true }));
                            setGenerationProgress(null);
                            setIsLoading(false);
                            newEventSource.close();
                            setEventSource(null);
                            break;

                        default:
                            console.log("[SSE] Unhandled message type:", data.type);
                            break;
                    }
                } catch (error) {
                    console.error("[SSE] Message parsing error:", error, "Raw data:", event.data);
                }
            };

            newEventSource.onerror = (error) => {
                console.error("[SSE] Connection error details:", {
                    readyState: newEventSource.readyState,
                    error: error
                });
                
                if (newEventSource.readyState === EventSource.CLOSED) {
                    console.log("[SSE] Connection closed - completing current progress");
                    
                    // Gracefully complete the current request
                    if (generationProgress && kgData) {  // Check if kgData exists
                        // Force complete with current progress
                        setCompletedSteps(prev => ({ ...prev, generate: true }));
                        if (kgData.kg_id) {
                            setKgId(kgData.kg_id);
                            setKgIds(prev => [...prev, kgData.kg_id as string]);
                        }
                    }
                    
                    // Clean up
                    setIsLoading(false);
                    setGenerationProgress(null);
                    newEventSource.close();
                    setEventSource(null);
                }
            };

            setEventSource(newEventSource);

        } catch (error) {
            console.error("[SSE] Setup error:", error);
            setIsLoading(false);
            setGenerationProgress(null);
        }
    };

    const handleQueryResult = (result: QueryResponse) => {
        console.log("Query Result:", result);
        setQueryResponse(result);
    };

    const handleEnrichKG = async (): Promise<void> => {
        if (kgIds.length === 0) {
            console.warn("No knowledge graph IDs available for enrichment");
            return;
        }

        setIsLoading(true);
        console.log("Enriching KGs with IDs:", kgIds);
        
        try {
            const response = await enrichKG(kgIds);
            console.log("Enriched Knowledge Graph Response:", response);
            console.log("Enriched Nodes:", response.kg.nodes.length);
            console.log("Enriched Edges:", response.kg.edges.length);
            
            setKgData(response.kg);
            setKgId(response.kg_id);
            setSelectedNodes(new Set(response.kg.nodes.map(node => node.id)));
            setCompletedSteps(prev => ({ ...prev, enrich: true }));
        } catch (error) {
            console.error("Error enriching KGs:", error);
        } finally {
            setIsLoading(false);
        }
    };

    const handleQuerySubmit = async () => {
        if (!kgId || !debouncedQuery.trim() || !kgData) return;
        
        setIsLoading(true);
        try {
            console.log("Querying KG with ID:", kgId);
            console.log("Top N:", topN, "Num Hops:", numHops);
            const nodesToQuery = selectedNodes.size > 0 
                ? Array.from(selectedNodes)
                : kgData.nodes.map(node => node.id);

            const result = await queryKG(
                kgId,
                debouncedQuery,
                topN, // Use top N parameter
                numHops, // Use number of hops parameter
                nodesToQuery
            );
            handleQueryResult(result);
            setCompletedSteps(prev => ({ ...prev, query: true }));
        } catch (error) {
            console.error("Error querying KG:", error);
        } finally {
            setIsLoading(false);
        }
    };

    const getRelevantArticles = (nodeId: string | null): Article[] => {
        if (!kgData) return [];
        
        // If no nodes are explicitly selected or selectedNodes is empty, return all articles
        if (selectedNodes.size === 0) {
            return kgData.articles;
        }

        // Get unique articles from all selected nodes
        const selectedArticleUrls = new Set<string>();
        Array.from(selectedNodes).forEach(nodeId => {
            const node = kgData.nodes.find(node => node.id === nodeId);
            if (node) {
                node.articles.forEach(url => selectedArticleUrls.add(url));
            }
        });

        // Filter articles based on the collected URLs
        return kgData.articles.filter(article => 
            selectedArticleUrls.has(article.url)
        );
    };

    /* Main components */
    return (
        <Box sx={{
            display: "flex",
            minHeight: "100vh",
            width: "100%",
            bgcolor: "background.default",
            position: "relative",
            overflow: "hidden"
        }}>
            {/* Main Content Area */}
            <Box sx={{
                flexGrow: 1,
                width: "100%",
                height: "100vh",
                transition: "all 0.3s ease-in-out",
                marginRight: isRightPanelOpen ? "400px" : 0,
                marginLeft: isLeftPanelOpen ? "400px" : 0,
                display: "flex",
                flexDirection: "column",
                pt: 2,
                px: 8,
                overflow: "auto"
            }}>
                {/* Header */}
                <Box sx={{ 
                    display: "flex", 
                    justifyContent: "space-between", 
                    alignItems: "center",
                    mb: 2,
                    pb: 2,
                    borderBottom: 1,
                    borderColor: "divider"
                }}>
                    <IconButton 
                        onClick={() => setIsLeftPanelOpen(!isLeftPanelOpen)}
                        size="small"
                    >
                        <KeyboardDoubleArrowLeftIcon 
                            sx={{ 
                                transform: isLeftPanelOpen ? "rotate(0deg)" : "rotate(180deg)",
                                transition: "transform 0.3s"
                            }}
                        />
                    </IconButton>
                    <Typography variant="h6">
                        GRAPH & ASK
                    </Typography>
                    <IconButton 
                        onClick={() => setIsRightPanelOpen(!isRightPanelOpen)}
                        size="small"
                    >
                        <KeyboardDoubleArrowLeftIcon 
                            sx={{ 
                                transform: isRightPanelOpen ? "rotate(180deg)" : "rotate(0deg)",
                                transition: "transform 0.3s"
                            }}
                        />
                    </IconButton>
                </Box>

                <Drawer
                    anchor="left"
                    open={isLeftPanelOpen}
                    onClose={() => setIsLeftPanelOpen(false)}
                    variant="persistent"
                    PaperProps={{
                        sx: {
                            width: "400px",
                            position: "fixed",
                            height: "100%",
                            top: 0,
                            pt: 2,
                            px: 2,
                            transition: "transform 0.3s ease-in-out",
                            transform: isLeftPanelOpen ? "translateX(0)" : "translateX(-100%)",
                            overflowY: "auto",
                            // borderRight: 1,
                            // borderColor: "divider",
                            // mr: 2
                        }
                    }}
                >
                    <Box sx={{ 
                        display: "flex", 
                        justifyContent: "space-between", 
                        alignItems: "center",
                        mb: 2,
                        position: "sticky",
                        top: 0,
                        bgcolor: "background.paper",
                        zIndex: 1,
                        py: 1
                    }}>
                        <Typography variant="h6">How to Use</Typography>
                        <IconButton 
                            onClick={() => setIsLeftPanelOpen(false)}
                            size="small"
                        >
                            <CloseIcon fontSize="small" />
                        </IconButton>
                    </Box>
                    
                    <List sx={{ px: 0 }}>
                        <ListItem sx={{ mb: 3 }}>
                            <ListItemIcon>
                                <LooksOneIcon color="primary" />
                            </ListItemIcon>
                            <ListItemText 
                                primary="Generate Knowledge Graph"
                                secondary="Generate knowledge graphs from news articles. Use side panel on right to view news summaries and their associated knowledge graphs as they are fetched, processed, and generated."
                            />
                        </ListItem>
                        
                        <ListItem sx={{ mb: 3 }}>
                            <ListItemIcon>
                                <LooksTwoIcon color="primary" />
                            </ListItemIcon>
                            <ListItemText 
                                primary="Enrich Knowledge Graph"
                                secondary="Enrich with semantic information and combine all knowledge graphs. After enrichment, each node is enriched with the following semantic information: i) a network embedding, ii) a text embedding, and iii) a community assignment. This step may take some time if there are many nodes in the knowledge graph. An overall summary of the enriched knowledge graph is also generated. Use side panel on right to see overall summary. The context nodes section at the bottom will display all the nodes. Hover over these chips to see the node type and summary."
                            />
                        </ListItem>
                        
                        <ListItem>
                            <ListItemIcon>
                                <Looks3Icon color="primary" />
                            </ListItemIcon>
                            <ListItemText 
                                primary="Query Knowledge Graph"
                                secondary="Query the combined knowledge graph to retrieve relevant information, and generate answer. Use knowledge graph visualisation to include the nodes to include in initial context. If no nodes are selected, all nodes are used as initial context. The final context comes from the connected nodes (K-hop) and the top-N nodes based on the semantic similarity of the query to the network embeddings, text embeddings, and the communities of the graph. Use side panel on right to see the answer to the query and evidence."
                            />
                        </ListItem>
                    </List>
                </Drawer>

                <Box sx={{ mb: 4 }}>
                    <Stack direction="row" spacing={2} sx={{ justifyContent: 'space-between' }}>
                        {/* Generate Button */}
                        <Stack direction="column" alignItems="center" position="relative">
                            <IconButton 
                                onClick={handleGenerateKG} 
                                disabled={isLoading}
                            >
                                {completedSteps.generate ? <PanoramaFishEyeIcon /> : <AdjustIcon />}
                            </IconButton>
                            <Typography variant="caption" color="textSecondary" align="center">
                                {isLoading ? (
                                    generationProgress 
                                        ? `Generating (${generationProgress.current}/${generationProgress.total})`
                                        : "..."
                                ) : "Generate"}
                            </Typography>
                            {isLoading && (
                                <Box sx={{
                                    position: "absolute",
                                    top: "50%",
                                    left: "50%",
                                    transform: "translate(-50%, -100%)",
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                }}>
                                    <CircularProgress
                                        variant={generationProgress ? "determinate" : "indeterminate"}
                                        value={generationProgress 
                                            ? (generationProgress.current / generationProgress.total) * 100 
                                            : undefined}
                                        size={20}
                                    />
                                </Box>
                            )}
                        </Stack>

                        {/* Enrich Button */}
                        <Stack direction="column" alignItems="center" position="relative">
                            <IconButton 
                                onClick={handleEnrichKG} 
                                disabled={isLoading || !completedSteps.generate}
                            >
                                {completedSteps.enrich ? <PanoramaFishEyeIcon /> : <AdjustIcon />}
                            </IconButton>
                            <Typography variant="caption" color="textSecondary" align="center">
                                {isLoading ? (
                                    generationProgress 
                                        ? `Enriching (${generationProgress.current}/${generationProgress.total})`
                                        : "..."
                                ) : "Enrich"}
                            </Typography>
                            {isLoading && (
                                <Box sx={{
                                    position: "absolute",
                                    top: "50%",
                                    left: "50%",
                                    transform: "translate(-50%, -100%)",
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                }}>
                                    <CircularProgress
                                        variant={generationProgress ? "determinate" : "indeterminate"}
                                        value={generationProgress 
                                            ? (generationProgress.current / generationProgress.total) * 100 
                                            : undefined}
                                        size={20}
                                    />
                                </Box>
                            )}
                        </Stack>

                        {/* Query Button */}
                        <Stack direction="column" alignItems="center" position="relative">
                            <IconButton 
                                onClick={handleQuerySubmit} 
                                disabled={isLoading || !completedSteps.enrich}
                            >
                                {completedSteps.query ? <PanoramaFishEyeIcon /> : <AdjustIcon />}
                            </IconButton>
                            <Typography variant="caption" color="textSecondary" align="center">
                                {isLoading ? (
                                    generationProgress 
                                        ? `Querying (${generationProgress.current}/${generationProgress.total})`
                                        : "..."
                                ) : "Query"}
                            </Typography>
                            {isLoading && (
                                <Box sx={{
                                    position: "absolute",
                                    top: "50%",
                                    left: "50%",
                                    transform: "translate(-50%, -100%)",
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                }}>
                                    <CircularProgress
                                        variant={generationProgress ? "determinate" : "indeterminate"}
                                        value={generationProgress 
                                            ? (generationProgress.current / generationProgress.total) * 100 
                                            : undefined}
                                        size={20}
                                    />
                                </Box>
                            )}
                        </Stack>
                    </Stack>
                    <Divider sx={{ my: 2}} />
                    <Stack direction="row" spacing={1} alignItems="center">
                        <Autocomplete<CompanyOption, true>
                            multiple
                            fullWidth
                            options={companyOptions}
                            value={tickers.split(",")
                                .filter(t => t.trim())
                                .map(symbol => 
                                    companyOptions.find(co => co.symbol === symbol.trim()) || {
                                        symbol: symbol.trim(),
                                        name: "",
                                        sector: ""
                                    }
                                )}
                            onChange={(_, newValue) => {
                                const limitedValue = newValue.slice(0, MAX_TICKERS);
                                setTickers(limitedValue.map(v => v.symbol).join(","));
                            }}
                            getOptionLabel={(option) => `${option.symbol} - ${option.name}`}
                            renderInput={(params) => (
                                <TextField
                                    {...params}
                                    label="Enter stock tickers (Max 3)"
                                    placeholder="Search by symbol or company name"
                                    size="small"
                                    margin="none"
                                />
                            )}
                            renderTags={(value) =>
                                value.map((option) => (
                                    <Chip
                                        key={option.symbol}
                                        size="small"
                                        label={option.symbol}
                                    />
                                ))
                            }
                            size="small"
                            sx={{ 
                                flexGrow: 1,
                                '& .MuiOutlinedInput-root': {
                                    height: '40px'
                                }
                            }}
                        />
                        <TextField
                            label="Window"
                            type="number"
                            value={window}
                            onChange={(e) => setWindow(Math.max(1, Math.min(MAX_WINDOW, parseInt(e.target.value) || 1)))}
                            size="small"
                            margin="none"
                            slotProps={{
                                input: {
                                    inputProps: {   
                                        min: 1,
                                        max: MAX_WINDOW
                                    }
                                }
                            }}
                            sx={{ 
                                width: "80px",
                                minWidth: "80px"
                            }}
                        />
                        <TextField
                            label="Limit"
                            type="number"
                            value={limit}
                            onChange={(e) => setLimit(Math.max(1, Math.min(MAX_ARTICLES, parseInt(e.target.value) || 1)))}
                            size="small"
                            margin="none"
                            slotProps={{
                                input: {
                                    inputProps: {
                                        min: 1,
                                        max: MAX_ARTICLES
                                    }
                                }
                            }}
                            sx={{ 
                                width: "80px",
                                minWidth: "80px"
                            }}
                        />
                    </Stack>

                    {completedSteps.enrich && (
                        <Box sx={{ mt: 4 }}>
                            <Stack direction="row" spacing={2} sx={{ width: "100%" }}>
                                <TextField
                                    fullWidth
                                    label="Enter your query"
                                    value={localQuery}
                                    onChange={(e) => setLocalQuery(e.target.value)}
                                    placeholder="Ask a question about the knowledge graph..."
                                    multiline
                                    rows={1}
                                    disabled={isLoading}
                                    sx={{
                                        "& .MuiOutlinedInput-root": {
                                            borderRadius: "8px",
                                        }
                                    }}
                                />
                                <TextField
                                    label="Top N"
                                    type="number"
                                    value={topN}
                                    onChange={(e) => setTopN(Math.max(1, parseInt(e.target.value) || 1))}
                                    margin="normal"
                                    sx={{ width: "150px" }} // Adjust width as needed
                                />
                                <TextField
                                    label="Hops"
                                    type="number"
                                    value={numHops}
                                    onChange={(e) => setNumHops(Math.max(1, parseInt(e.target.value) || 1))}
                                    margin="normal"
                                    sx={{ width: "150px" }} // Adjust width as needed
                                />
                                {/* <Button
                                    variant="contained"
                                    onClick={handleQuerySubmit}
                                    disabled={isLoading || !localQuery.trim()}
                                    sx={{
                                        height: "56px",
                                        alignSelf: "flex-end",
                                        minWidth: "120px"
                                    }}
                                >
                                    {isLoading ? (
                                        <CircularProgress size={24} color="inherit" />
                                    ) : (
                                        "Query"
                                    )}
                                </Button> */}
                            </Stack>
                        </Box>
                    )}
                    {kgData && (
                        <Box sx={{ mt: 4 }}>
                            <Box sx={{ 
                                display: "flex",
                                gap: 2,
                                height: "600px",
                            }}>
                                <Box sx={{ 
                                    flex: 1,
                                    display: "flex",
                                    flexDirection: "column",
                                    height: "100%",
                                    overflow: "hidden"
                                }}>
                                    <MemoizedGraphVisualization 
                                        data={kgData}
                                        selectedNodes={selectedNodes}
                                        onNodeClick={(nodeId: string, isMultiSelect?: boolean) => {
                                            if (isMultiSelect) {
                                                const newSelectedNodes = new Set(selectedNodes);
                                                if (newSelectedNodes.has(nodeId)) {
                                                    newSelectedNodes.delete(nodeId);
                                                    setSelectedNodeId(null);
                                                } else {
                                                    newSelectedNodes.add(nodeId);
                                                    setSelectedNodeId(nodeId);
                                                }
                                                setSelectedNodes(newSelectedNodes);
                                            } else {
                                                setSelectedNodes(new Set([nodeId]));
                                                setSelectedNodeId(nodeId);
                                            }
                                        }}
                                        onSelectionClear={() => {
                                            setSelectedNodes(new Set());
                                            setSelectedNodeId(null);
                                        }}
                                        completedSteps={completedSteps}
                                    />
                                </Box>
                            </Box>
                                <Box sx={{ mt: 2, mb: 2 }}>
                                <Box sx={{ 
                                    display: "flex", 
                                    justifyContent: "space-between", 
                                    alignItems: "center",
                                    mb: 1
                                }}>
                                    <Typography variant="caption" color="textSecondary" sx={{ mb: 1, display: "block" }}>
                                        Context Nodes ({selectedNodes.size})
                                    </Typography>
                                    <Box>
                                    {selectedNodes.size > 0 && (
                                            <Button 
                                                size="small" 
                                                variant="text" 
                                                color="primary"
                                                onClick={() => {
                                                    setSelectedNodes(new Set());
                                                    setSelectedNodeId(null);
                                                }}
                                                startIcon={<RemoveCircleIcon fontSize="small" />}
                                            >
                                                Clear All
                                            </Button>
                                        )}
                                        <Button 
                                            size="small" 
                                            variant="text" 
                                            color="primary"
                                            onClick={() => {
                                                const allNodeIds = kgData?.nodes.map(node => node.id) || [];
                                                setSelectedNodes(new Set(allNodeIds));
                                            }}
                                            startIcon={<AddCircleIcon fontSize="small" />} 
                                        >
                                            Select All
                                            </Button>
                                        </Box>
                                    </Box>
                                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                                        {Array.from(selectedNodes).map((nodeId) => {
                                            const node = kgData?.nodes.find(n => n.id === nodeId);
                                            // const tooltipTitle = node ? `Type: ${node.type}Summary: ${node.summary}` : "";

                                            return (
                                                <Tooltip title={
                                                    <React.Fragment>
                                                        <div>Type: {node?.type}</div>
                                                        <div>Summary: {node?.summary}</div>
                                                    </React.Fragment>
                                                } key={nodeId}
                                                PopperProps={{
                                                    sx: {
                                                        '& div': {
                                                            padding: "4px"
                                                        }
                                                    }
                                                }}  
                                                >
                                                    <Chip
                                                        label={nodeId}
                                                        onDelete={() => {
                                                            const newSelectedNodes = new Set(selectedNodes);
                                                            newSelectedNodes.delete(nodeId);
                                                            setSelectedNodes(newSelectedNodes);
                                                        }}
                                                        size="small"
                                                        sx={{ mb: 1 }}
                                                    />
                                                </Tooltip>
                                            );
                                        })}
                                    </Stack>
                                </Box>
                        </Box>
                    )}
                    <Drawer
                        anchor="right"
                        open={isRightPanelOpen}
                        onClose={() => setIsRightPanelOpen(false)}
                        variant="persistent"
                        PaperProps={{
                            sx: {
                                width: "400px",
                                position: "fixed",
                                height: "100%",
                                top: 0,
                                pt: 2,
                                px: 2,
                                transition: "transform 0.3s ease-in-out",
                                transform: isRightPanelOpen ? "translateX(0)" : "translateX(100%)",
                                overflowY: "auto"
                            }
                        }}
                    >
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant="h6">Additional Information</Typography>
                            <IconButton onClick={() => setIsRightPanelOpen(false)}>
                                <CloseIcon />
                            </IconButton>
                        </Box>
                        <ArticlePanel
                            articles={getRelevantArticles(selectedNodeId)}
                            selectedNodeId={selectedNodeId}
                            onClose={() => setSelectedNodeId(null)}
                            kgSummary={kgData?.summary}
                            isEnriched={completedSteps.enrich}
                            queryResponse={queryResponse}
                            currentQuery={query}
                        />
                    </Drawer>
                </Box>
            </Box>
        </Box>
    );
};

export default App;
