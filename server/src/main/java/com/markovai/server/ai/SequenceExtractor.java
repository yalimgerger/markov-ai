package com.markovai.server.ai;

public interface SequenceExtractor {
    /**
     * Extracts a sequence of states from a binary image.
     * 
     * @param binaryImage 28x28 binary image (values 0 or 1)
     * @return 1D array of states
     */
    int[] extractSequence(int[][] binaryImage);
}
