package com.markovai.server.ai;

import java.util.List;

public interface MultiSequenceExtractor {
    /**
     * Extracts multiple sequences of states from a binary image.
     * 
     * @param binaryImage 28x28 binary image
     * @return List of 1D integer arrays
     */
    List<int[]> extractSequences(int[][] binaryImage);
}
