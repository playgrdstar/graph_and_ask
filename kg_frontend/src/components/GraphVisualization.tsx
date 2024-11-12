import React, { useRef, useState, useEffect } from "react";
import CytoscapeComponent from "react-cytoscapejs";
import { Core } from "cytoscape";
import { 
    Box, 
    Slider, 
    Typography, 
    Stack,
    IconButton,
    Paper
} from "@mui/material";
import { KnowledgeGraph, KGNode, KGEdge } from "../types/api.types";
import { cosineSimilarity } from "../utils/math";
import CenterFocusStrongIcon from '@mui/icons-material/CenterFocusStrong';

interface GraphVisualizationProps {
    data: KnowledgeGraph;
    selectedNodes: Set<string>;
    onNodeClick: (nodeId: string, isMultiSelect?: boolean) => void;
    onSelectionClear: () => void;
    completedSteps: {
        enrich: boolean;
    };
}

const GraphVisualization: React.FC<GraphVisualizationProps> = ({
    data,
    selectedNodes,
    onNodeClick,
    onSelectionClear,
    completedSteps,
}) => {
    const cyRef = useRef<Core | null>(null);
    const [networkThreshold, setNetworkThreshold] = useState<number>(0.8);
    const [textThreshold, setTextThreshold] = useState<number>(0.8);

    // Add ref to track previous data for diffing
    const prevDataRef = useRef<KnowledgeGraph | null>(null);

    // Update getSimilarNodes to use separate thresholds
    const getSimilarNodes = (node: KGNode, nodes: KGNode[]) => {
        if (!node.network_embedding || !node.text_embedding) return new Set<string>();
        
        return new Set(nodes.filter(n => {
            if (!n.network_embedding || !n.text_embedding) return false;
            
            const networkSimilarity = cosineSimilarity(
                node.network_embedding as number[],
                n.network_embedding as number[]
            );
            const textSimilarity = cosineSimilarity(
                node.text_embedding as number[],
                n.text_embedding as number[]
            );
            
            return n.id !== node.id && 
                (networkSimilarity > networkThreshold || 
                 textSimilarity > textThreshold);
        }).map(n => n.id));
    };

    // Function to get directly connected nodes
    const getConnectedNodes = (nodeId: string, edges: KGEdge[]) => {
        return new Set(edges
            .filter(edge => edge.source === nodeId || edge.target === nodeId)
            .map(edge => edge.source === nodeId ? edge.target : edge.source));
    };

    // Function to get nodes in the same community
    const getCommunityNodes = (node: KGNode, nodes: KGNode[]) => {
        if (node.community === null) return new Set<string>();
        
        return new Set(nodes
            .filter(n => 
                n.id !== node.id && 
                n.community !== null && 
                n.community === node.community
            )
            .map(n => n.id));
    };

    // Convert graph data to Cytoscape elements with validation
    const getElements = () => {
        try {
            // Create a Set of valid node IDs for quick lookup
            const validNodeIds = new Set(data.nodes.map(node => node.id));

            // Filter and map nodes
            const nodes = data.nodes.map(node => ({
                data: {
                    id: node.id,
                    label: node.id,
                    width: 15,
                    height: 15,
                }
            }));

            // Filter and map edges, excluding those with invalid source/target
            const validEdges = data.edges.filter(edge => {
                const isValid = validNodeIds.has(edge.source) && validNodeIds.has(edge.target);
                if (!isValid) {
                    // Log warning but don't throw error
                    console.warn(
                        "Invalid edge found:",
                        {
                            source: edge.source,
                            target: edge.target,
                            label: edge.label,
                            validSource: validNodeIds.has(edge.source),
                            validTarget: validNodeIds.has(edge.target)
                        }
                    );
                }
                return isValid;
            });

            const edges = validEdges.map(edge => ({
                data: {
                    id: `${edge.source}-${edge.target}`,
                    source: edge.source,
                    target: edge.target,
                    label: edge.label
                }
            }));

            // Debug logging
            console.debug("Graph elements:", {
                totalNodes: nodes.length,
                totalEdges: edges.length,
                validNodeIds: Array.from(validNodeIds),
            });

            return [...nodes, ...edges];
        } catch (error) {
            console.error("Error in getElements:", error);
            // Return at least the nodes if there's an error with edges
            return data.nodes.map(node => ({
                data: {
                    id: node.id,
                    label: node.id,
                    width: 15,
                    height: 15,
                }
            }));
        }
    };

    const handleNodeSelection = (nodeId: string, isMultiSelect: boolean = false) => {
        if (!cyRef.current) return;
        const cy = cyRef.current;

        // Store current viewport position and zoom
        const zoom = cy.zoom();
        const pan = cy.pan();

        console.log("Node selected:", {
            nodeId,
            isMultiSelect,
            originalNode: data.nodes.find(n => n.id === nodeId)
        });
        
        const selectedNode = data.nodes.find(n => n.id === nodeId);
        if (!selectedNode) return;

        // Update selected nodes set
        const newSelectedNodes = new Set(isMultiSelect ? selectedNodes : []);
        if (selectedNodes.has(nodeId) && isMultiSelect) {
            newSelectedNodes.delete(nodeId);
        } else {
            newSelectedNodes.add(nodeId);
        }

        // Reset all nodes to default style
        cy.nodes().removeClass("connected similar-embedding same-community selected");
        cy.nodes().style({
            "background-color": "#666",
            "border-width": 2,
            "border-color": "#666",
            "opacity": 0.3
        });

        // Style for all selected nodes and their related nodes
        newSelectedNodes.forEach(selectedId => {
            const node = data.nodes.find(n => n.id === selectedId);
            if (!node) return;

            const connectedNodes = getConnectedNodes(selectedId, data.edges);
            const communityNodes = getCommunityNodes(node, data.nodes);
            const similarNodes = getSimilarNodes(node, data.nodes);

            // Apply styles
            cy.nodes().forEach(cyNode => {
                const id = cyNode.id();
                if (newSelectedNodes.has(id)) {
                    cyNode.addClass("selected");
                    cyNode.style({
                        "background-color": "lightblue",
                        "border-color": "lightblue",
                        "opacity": 1
                    });
                } else if (connectedNodes.has(id)) {
                    cyNode.addClass("connected");
                    cyNode.style({
                        "background-color": "lightblue",
                        "border-color": "lightblue",
                        "opacity": 0.8
                    });
                } else if (communityNodes.has(id)) {
                    cyNode.addClass("same-community");
                    cyNode.style({
                        "background-color": "orange",
                        "border-color": "orange",
                        "opacity": 0.8
                    });
                } else if (similarNodes.has(id)) {
                    cyNode.addClass("similar-embedding");
                    cyNode.style({
                        "background-color": "pink",
                        "border-color": "pink",
                        "opacity": 0.8
                    });
                }
            });
        });

        // Highlight edges connected to any selected node
        cy.edges().style({ "opacity": 0.5 });
        cy.edges().filter(edge => 
            newSelectedNodes.has(edge.source().id()) || 
            newSelectedNodes.has(edge.target().id())
        ).style({ "opacity": 1 });

        // Call onNodeClick with the latest selected node
        onNodeClick?.(nodeId, isMultiSelect);

        // Restore viewport position and zoom after style updates
        cy.viewport({
            zoom: zoom,
            pan: pan
        });
    };

    const handleCyInit = (cy: Core): void => {
        cyRef.current = cy;

        cy.style([
            {
                selector: 'node',
                style: {
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'background-color': '#666',
                    'border-width': 2,
                    'border-color': '#666',
                    'width': 'data(width)',
                    'height': 'data(height)',
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 1,
                    'line-color': '#ccc',
                    'target-arrow-color': '#ccc',
                    'target-arrow-shape': 'none',
                    'curve-style': 'bezier',
                    'label': 'data(label)',
                    'font-size': '10px',
                    'text-rotation': 'autorotate',
                }
            }
        ]);

        cy.on("tap", "node", (event) => {
            const nodeId = event.target.id();
            handleNodeSelection(nodeId, event.originalEvent.shiftKey);
        });

        cy.on("tap", function(event) {
            if (event.target === cy) {
                // Clicked on background - reset styles and clear selection
                cy.nodes().removeClass("connected similar-embedding same-community selected");
                cy.nodes().style({
                    "background-color": "#666",
                    "border-width": 2,
                    "border-color": "#666",
                    "opacity": 1
                });
                cy.edges().style({ "opacity": 1 });
                onSelectionClear();
            }
        });
    };

    // Add error boundary for Cytoscape component
    const handleError = (error: any) => {
        console.error("Error in graph visualization:", error);
        // You could set an error state here if you want to show an error message to the user
    };

    // Validate graph data before rendering
    const validateGraphData = () => {
        if (!data || !data.nodes || !data.edges) {
            throw new Error("Invalid graph data structure");
        }

        // Check nodes
        const nodeIssues = data.nodes.filter(node => !node.id);
        if (nodeIssues.length > 0) {
            console.warn("Nodes with missing IDs:", nodeIssues);
        }

        // Check edges
        const edgeIssues = data.edges.filter(edge => !edge.source || !edge.target);
        if (edgeIssues.length > 0) {
            console.warn("Edges with missing source/target:", edgeIssues);
        }

        // Check for duplicate node IDs
        const nodeIds = new Set<string>();
        const duplicateNodes = data.nodes.filter(node => {
            if (nodeIds.has(node.id)) {
                return true;
            }
            nodeIds.add(node.id);
            return false;
        });

        if (duplicateNodes.length > 0) {
            console.warn("Duplicate node IDs found:", duplicateNodes);
        }
    };

    const handleResetView = () => {
        if (!cyRef.current) return;
        cyRef.current.fit(); // Fits and centers the graph
        cyRef.current.zoom({ // Optional: set a specific zoom level
            level: 1.25,
            position: { x: 0, y: 0 }
        });
    };

    // Optimize the update effect
    useEffect(() => {
        console.log("[GraphViz] Data changed:", {
            timestamp: new Date().toISOString(),
            nodeCount: data.nodes.length,
            edgeCount: data.edges.length
        });

        if (!cyRef.current) return;

        const cy = cyRef.current;
        const prevData = prevDataRef.current;
        
        // Step 1: Add all new nodes first
        const newNodes = data.nodes.filter(node => 
            !prevData?.nodes.some(n => n.id === node.id)
        );

        if (newNodes.length > 0) {
            const nodeElements = newNodes.map(node => ({
                data: {
                    id: node.id,
                    label: node.id,
                    width: 15,
                    height: 15,
                }
            }));
            cy.add(nodeElements);
        }

        // Step 2: After nodes are added, process edges
        const existingNodeIds = new Set(cy.nodes().map(n => n.id()));
        const validNewEdges = data.edges.filter(edge => 
            !prevData?.edges.some(e => 
                e.source === edge.source && e.target === edge.target
            ) &&
            existingNodeIds.has(edge.source) && 
            existingNodeIds.has(edge.target)
        );

        if (validNewEdges.length > 0) {
            const edgeElements = validNewEdges.map(edge => ({
                data: {
                    id: `${edge.source}-${edge.target}`,
                    source: edge.source,
                    target: edge.target,
                    label: edge.label
                }
            }));
            cy.add(edgeElements);
        }

        // Step 3: Remove obsolete elements
        if (prevData) {
            // Remove obsolete edges first
            const removedEdges = prevData.edges.filter(edge => 
                !data.edges.some(e => 
                    e.source === edge.source && e.target === edge.target
                )
            );
            
            removedEdges.forEach(edge => {
                cy.$(`edge[source = "${edge.source}"][target = "${edge.target}"]`).remove();
            });

            // Then remove obsolete nodes
            const removedNodeIds = prevData.nodes
                .filter(node => !data.nodes.some(n => n.id === node.id))
                .map(node => node.id);
            
            if (removedNodeIds.length > 0) {
                cy.nodes(`[id = "${removedNodeIds.join('"], [id = "')}"]`).remove();
            }
        }

        // Step 4: Only run layout if there were changes
        if (newNodes.length > 0 || validNewEdges.length > 0) {
            cy.layout({ 
                name: "cose", 
                animate: false,
                randomize: false,
                componentSpacing: 40,
                nodeRepulsion: () => 400000
            }).run();
        }

        // Update reference to current data
        prevDataRef.current = data;
    }, [data]);

    // Wrap the render in try-catch
    try {
        validateGraphData();
        
        return (
            <Box>
                <Stack spacing={1} sx={{ mt: 2, mb: 2, ml: 4, mr: 4 }}>
                    {/* Network Similarity Slider */}
                    {completedSteps.enrich && (
                    <Box sx={{ 
                        display: "flex", 
                        alignItems: "center", 
                        gap: 1
                    }}>
                        <Typography 
                            variant="body2" 
                            color="textSecondary"
                            sx={{ minWidth: "100px" }}
                        >
                            Network Similarity:
                        </Typography>
                        <Slider
                            value={networkThreshold}
                            onChange={(_, value) => {
                                setNetworkThreshold(value as number);
                                // Reapply highlighting if a node is selected
                                const selectedNode = cyRef.current?.nodes('.selected').first();
                                if (selectedNode) {
                                    handleNodeSelection(selectedNode.id());
                                }
                            }}
                            min={0}
                            max={1}
                            step={0.1}
                            valueLabelDisplay="off"
                            sx={{ 
                                flex: 1,
                                '& .MuiSlider-thumb': {
                                    backgroundColor: 'gray',
                                },
                                '& .MuiSlider-track': {
                                    backgroundColor: 'gray',
                                },
                                '& .MuiSlider-rail': {
                                    backgroundColor: 'gray',
                                }
                            }}
                        />
                        <Typography variant="body2" sx={{ minWidth: "40px", textAlign: "right" }}>
                            {networkThreshold.toFixed(2)}
                        </Typography>
                        </Box>
                    )}

                    {/* Text Similarity Slider */}
                    {completedSteps.enrich && (
                        <Box sx={{ 
                        display: "flex", 
                        alignItems: "center", 
                        gap: 1
                    }}>
                        <Typography 
                            variant="body2" 
                            color="textSecondary"
                            sx={{ minWidth: "100px" }}
                        >
                            Text Similarity:
                        </Typography>
                        <Slider
                            value={textThreshold}
                            onChange={(_, value) => {
                                setTextThreshold(value as number);
                                // Reapply highlighting if a node is selected
                                const selectedNode = cyRef.current?.nodes('.selected').first();
                                if (selectedNode) {
                                    handleNodeSelection(selectedNode.id());
                                }
                            }}
                            min={0}
                            max={1}
                            step={0.1}
                            valueLabelDisplay="off"
                            sx={{ 
                                flex: 1,
                                '& .MuiSlider-thumb': {
                                    backgroundColor: 'gray',
                                },
                                '& .MuiSlider-track': {
                                    backgroundColor: 'gray',
                                },
                                '& .MuiSlider-rail': {
                                    backgroundColor: 'gray',
                                }
                            }}
                        />
                        <Typography variant="body2" sx={{ minWidth: "40px", textAlign: "right" }}>
                            {textThreshold.toFixed(2)}
                            </Typography>
                        </Box>
                    )}
                </Stack>

                <Box sx={{ 
                    border: "1px solid #ddd",
                    borderRadius: "8px",
                    overflow: "hidden",
                    height: "600px",
                    position: "relative"
                }}>
                    {completedSteps.enrich && <Legend />}
                    <IconButton
                        onClick={handleResetView}
                        sx={{
                            position: "absolute",
                            right: 16,
                            top: 16,
                            zIndex: 1,
                            backgroundColor: "background.paper",
                            boxShadow: 1,
                            "&:hover": {
                                backgroundColor: "background.default"
                            }
                        }}
                        size="small"
                    >
                        <CenterFocusStrongIcon />
                    </IconButton>

                    <CytoscapeComponent
                        elements={getElements()}
                        style={{ width: "100%", height: "100%" }}
                        cy={handleCyInit}
                        layout={{ name: "cose", animate: false }}
                    />
                </Box>
            </Box>
        );
    } catch (error) {
        console.error("Failed to render graph:", error);
        return (
            <Box sx={{ 
                p: 3, 
                border: "1px solid #ff0000", 
                borderRadius: "8px",
                bgcolor: "#fff5f5"
            }}>
                <Typography color="error" variant="h6">
                    Error Rendering Graph
                </Typography>
                <Typography color="error" variant="body2">
                    There was an error rendering the graph visualization. 
                    Please check the console for more details.
                </Typography>
            </Box>
        );
    }
};

const Legend = () => (
    <Paper
        elevation={2}
        sx={{
            position: "absolute",
            left: 16,
            top: 16,
            zIndex: 1,
            padding: 1.5,
            backgroundColor: "background.paper",
        }}
    >
        <Typography variant="body2" fontWeight="bold" sx={{ mb: 1 }}>
            Legend
        </Typography>
        <Stack spacing={1}>
            {[
                { color: "lightblue", label: "Selected or Connected Nodes" },
                { color: "pink", label: "Similar Nodes (Network or Text)" },
                { color: "orange", label: "Same Community" },
            ].map(({ color, label }) => (
                <Box key={color} sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Box
                        sx={{
                            width: 12,
                            height: 12,
                            backgroundColor: color,
                            borderRadius: "50%"
                        }}
                    />
                    <Typography variant="body2">{label}</Typography>
                </Box>
            ))}
        </Stack>
    </Paper>
);

export default GraphVisualization;