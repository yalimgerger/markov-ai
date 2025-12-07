package com.markovai.server.ai;

import java.util.HashMap;
import java.util.Map;

public class DigitPatch4x4UnigramModel {

    // For each digit 0..9, map: patchSymbol -> count
    private final Map<Integer, Integer>[] countsPerDigit;

    // Total patch counts per digit
    private final int[] totalPatchesPerDigit;

    // After finalize: patchSymbol -> log P(patch | digit)
    private final Map<Integer, Double>[] logProbPerDigit;

    // Log probability for unseen symbols per digit
    private final double[] logProbUnseen;

    // Laplace smoothing constant
    private final double alpha = 1.0;

    @SuppressWarnings("unchecked")
    public DigitPatch4x4UnigramModel() {
        countsPerDigit = new HashMap[10];
        totalPatchesPerDigit = new int[10];
        logProbPerDigit = new HashMap[10];
        logProbUnseen = new double[10];

        for (int d = 0; d < 10; d++) {
            countsPerDigit[d] = new HashMap<>();
            logProbPerDigit[d] = new HashMap<>();
        }
    }

    public void trainOnImage(int digitLabel, int[][] binaryImage) {
        // Iterate over non-overlapping 4x4 patches on 28x28 grid
        // 7x7 = 49 patches
        for (int r = 0; r < 7; r++) {
            for (int c = 0; c < 7; c++) {
                int symbol = encodePatch(binaryImage, r * 4, c * 4);

                countsPerDigit[digitLabel].merge(symbol, 1, Integer::sum);
                totalPatchesPerDigit[digitLabel]++;
            }
        }
    }

    public static int encodePatch(int[][] binaryImage, int startRow, int startCol) {
        int symbol = 0;
        for (int pr = 0; pr < 4; pr++) {
            for (int pc = 0; pc < 4; pc++) {
                symbol <<= 1;
                // Safety check for bounds, though 28x28 fits 4x4 perfectly (7 steps)
                if (startRow + pr < binaryImage.length && startCol + pc < binaryImage[0].length) {
                    if (binaryImage[startRow + pr][startCol + pc] != 0) {
                        symbol |= 1;
                    }
                }
            }
        }
        return symbol;
    }

    public double logProbForSymbol(int digit, int symbol) {
        Double lp = logProbPerDigit[digit].get(symbol);
        if (lp == null) {
            return logProbUnseen[digit];
        }
        return lp;
    }

    public void finalizeProbabilities() {
        // Vocabulary size is 65536 (all possible 4x4 binary patterns)
        double vocabSize = 65536.0;

        for (int d = 0; d < 10; d++) {
            int total = totalPatchesPerDigit[d];
            double denom = total + alpha * vocabSize;

            // P(unseen) = alpha / denom
            logProbUnseen[d] = Math.log(alpha / denom);

            for (Map.Entry<Integer, Integer> entry : countsPerDigit[d].entrySet()) {
                int count = entry.getValue();
                double p = (count + alpha) / denom;
                logProbPerDigit[d].put(entry.getKey(), Math.log(p));
            }
        }
    }

    public double[] sumLogLikelihoodsForImage(int[][] binaryImage) {
        double[] sumLogL = new double[10]; // Init to 0.0

        for (int r = 0; r < 7; r++) {
            for (int c = 0; c < 7; c++) {
                int symbol = encodePatch(binaryImage, r * 4, c * 4);

                for (int d = 0; d < 10; d++) {
                    Double lp = logProbPerDigit[d].get(symbol);
                    if (lp == null) {
                        lp = logProbUnseen[d];
                    }
                    sumLogL[d] += lp;
                }
            }
        }
        return sumLogL;
    }

    public int getNumPatchesPerImage() {
        return 49;
    }
}
