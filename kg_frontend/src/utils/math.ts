/**
 * Calculate cosine similarity between two vectors
 * @param a First vector
 * @param b Second vector
 * @returns Cosine similarity value between -1 and 1
 */
export const cosineSimilarity = (a: number[], b: number[]): number => {
    if (!a || !b || a.length !== b.length) return 0;
    
    try {
        const dotProduct = a.reduce((sum, _, i) => sum + a[i] * b[i], 0);
        const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
        const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
        
        if (magnitudeA === 0 || magnitudeB === 0) return 0;
        
        return dotProduct / (magnitudeA * magnitudeB);
    } catch (error) {
        console.error("Error calculating cosine similarity:", error);
        return 0;
    }
};