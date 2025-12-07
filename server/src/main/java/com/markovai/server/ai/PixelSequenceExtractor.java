package com.markovai.server.ai;

public class PixelSequenceExtractor implements SequenceExtractor {

    // Explicitly defining dimensions for validation, similar to original logic
    private static final int IMAGE_SIZE = 28;
    private static final int SEQ_LENGTH = IMAGE_SIZE * IMAGE_SIZE;

    @Override
    public int[] extractSequence(int[][] binaryImage) {
        int[] seq = new int[SEQ_LENGTH];
        int idx = 0;
        for (int r = 0; r < IMAGE_SIZE; r++) {
            for (int c = 0; c < IMAGE_SIZE; c++) {
                seq[idx++] = binaryImage[r][c];
            }
        }
        return seq;
    }
}
