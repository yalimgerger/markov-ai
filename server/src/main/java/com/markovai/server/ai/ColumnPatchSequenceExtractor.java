package com.markovai.server.ai;

import java.util.ArrayList;
import java.util.List;

public class ColumnPatchSequenceExtractor implements MultiSequenceExtractor {

    private static final int PATCH_SIZE = 2;
    private static final int GRID_SIZE = 14;

    @Override
    public List<int[]> extractSequences(int[][] binaryImage) {
        List<int[]> sequences = new ArrayList<>(GRID_SIZE);

        for (int c = 0; c < GRID_SIZE; c++) {
            int[] colSeq = new int[GRID_SIZE];
            for (int r = 0; r < GRID_SIZE; r++) {
                colSeq[r] = getPatchSymbol(binaryImage, r, c);
            }
            sequences.add(colSeq);
        }
        return sequences;
    }

    private int getPatchSymbol(int[][] binaryImage, int r, int c) {
        int pr = r * PATCH_SIZE;
        int pc = c * PATCH_SIZE;

        int b3 = binaryImage[pr][pc]; // TL
        int b2 = binaryImage[pr][pc + 1]; // TR
        int b1 = binaryImage[pr + 1][pc]; // BL
        int b0 = binaryImage[pr + 1][pc + 1]; // BR

        return (b3 * 8) + (b2 * 4) + (b1 * 2) + b0;
    }
}
