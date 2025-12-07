package com.markovai.server.ai;

public class PatchSequenceExtractor implements SequenceExtractor {

    private static final int IMAGE_SIZE = 28;
    private static final int PATCH_SIZE = 2;
    private static final int GRID_SIZE = IMAGE_SIZE / PATCH_SIZE; // 14
    private static final int SEQ_LENGTH = GRID_SIZE * GRID_SIZE; // 196

    @Override
    public int[] extractSequence(int[][] binaryImage) {
        int[] seq = new int[SEQ_LENGTH];
        int idx = 0;

        for (int r = 0; r < GRID_SIZE; r++) {
            for (int c = 0; c < GRID_SIZE; c++) {
                // Patch coordinates (top-left)
                int pr = r * PATCH_SIZE;
                int pc = c * PATCH_SIZE;

                // Extract 4 bits
                // Top-Left -> bit 3 (8)
                // Top-Right -> bit 2 (4)
                // Bottom-Left -> bit 1 (2)
                // Bottom-Right -> bit 0 (1)

                int b3 = binaryImage[pr][pc];
                int b2 = binaryImage[pr][pc + 1];
                int b1 = binaryImage[pr + 1][pc];
                int b0 = binaryImage[pr + 1][pc + 1];

                int symbol = (b3 * 8) + (b2 * 4) + (b1 * 2) + b0;
                seq[idx++] = symbol;
            }
        }
        return seq;
    }
}
