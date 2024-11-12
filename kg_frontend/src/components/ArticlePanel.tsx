import React from "react";
import {
    Box,
    Typography,
    Paper,
    Link,
    List,
    ListItem,
    ListItemText,
    Divider,
    IconButton,
    Stack,
    Chip,
    ListItemIcon
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import RefreshIcon from '@mui/icons-material/Refresh';
import CircleIcon from '@mui/icons-material/Circle';
import LinkIcon from '@mui/icons-material/Link';
import { QueryResponse } from "../types/api.types";
import { decodeHtmlEntities } from '../utils/string';

interface Article {
    title: string;
    summary: string;
    url: string;
}

/**
 * Props for the ArticlePanel component
 */
interface ArticlePanelProps {
    /** List of articles to display */
    articles: Article[];
    /** Callback function to close the panel */
    onClose: () => void;
    /** ID of the selected node, null if no node is selected */
    selectedNodeId: string | null;
    /** Overall summary of the knowledge graph */
    kgSummary?: string;
    /** Whether the graph has been enriched */
    isEnriched: boolean;
    /** Query response to display */
    queryResponse: QueryResponse | null;
    /** Current query to display */
    currentQuery: string;
}

const ArticlePanel: React.FC<ArticlePanelProps> = ({
    articles,
    onClose,
    selectedNodeId,
    kgSummary,
    isEnriched,
    queryResponse,
    currentQuery
}) => {
    return (
        <Box sx={{ width: "400px", height: "100%", overflow: "auto" }}>
            {/* Query Results Section */}
            {queryResponse && (
                <Box sx={{ mb: 3, p: 2, backgroundColor: "#f5f5f5", borderRadius: 1 }}>
                    <Typography variant="subtitle1" gutterBottom>
                        Query Results
                    </Typography>
                    <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                        {currentQuery}
                    </Typography>
                    
                    {/* Answer */}
                    <Typography variant="body1">
                        {queryResponse.answer.answer}
                    </Typography>

                    {/* Key Entities */}
                    {queryResponse.answer.key_entities.length > 0 && (
                        <>
                            <Typography variant="subtitle1" gutterBottom>
                                Key Entities
                            </Typography>
                            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mb: 2 }}>
                                {queryResponse.answer.key_entities.map((entity: string, index: number) => (
                                    <Chip
                                        key={index}
                                        label={entity}
                                        size="small"
                                        sx={{ backgroundColor: "#e0e0e0" }}
                                    />
                                ))}
                            </Box>
                        </>
                    )}

                    {/* Evidence */}
                    {queryResponse.answer.evidence.length > 0 && (
                        <>
                            <Typography variant="subtitle1" gutterBottom>
                                Evidence
                            </Typography>
                            <List dense>
                                {queryResponse.answer.evidence.map((item: string, index: number) => (
                                    <ListItem key={index}>
                                        <ListItemIcon>
                                            <CircleIcon sx={{ fontSize: 8 }} />
                                        </ListItemIcon>
                                        <ListItemText primary={item} />
                                    </ListItem>
                                ))}
                            </List>
                        </>
                    )}

                </Box>
            )}

            {/* Existing Article Panel Content */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                {onClose && (
                    <IconButton onClick={onClose} size="small">
                        <RefreshIcon />
                    </IconButton>
                )}
            </Stack>

            {/* Show KG summary after enrichment */}
            {isEnriched && kgSummary && !selectedNodeId && (
                <Box sx={{ mb: 3, p: 2, bgcolor: "grey.50", borderRadius: 1 }}>
                    <Typography variant="subtitle2" gutterBottom>
                        Overall Summary
                    </Typography>
                    <Typography variant="body2">
                        {kgSummary}
                    </Typography>
                </Box>
            )}
            
            {/* Show articles section */}
            <Typography variant="subtitle1" sx={{ mb: 2 }}>
                {selectedNodeId ? "Related Articles" : "All Articles"}
            </Typography>
            
            {articles.length === 0 ? (
                <Typography variant="body2" sx={{ textAlign: "center", py: 2 }}>
                    No articles found for this node.
                </Typography>
            ) : (
                <List>
                    {articles.map((article, index) => (
                        <React.Fragment key={index}>
                            {index > 0 && <Divider sx={{ my: 2 }} />}
                            <ListItem sx={{ display: "flex", flexDirection: "column", alignItems: "flex-start" }}>
                                <Typography variant="subtitle2" sx={{ fontWeight: "bold", mb: 1 }}>
                                    {decodeHtmlEntities(article.title)}
                                </Typography>
                                <Typography variant="body2" sx={{ mb: 1 }}>
                                    {decodeHtmlEntities(article.summary)}
                                </Typography>
                                <Link
                                    href={article.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    sx={{ color: "primary.main" }}
                                >
                                    View Source
                                </Link>
                            </ListItem>
                        </React.Fragment>
                    ))}
                </List>
            )}
        </Box>
    );
};

export default ArticlePanel; 