package com.markovai.server.ai;

import org.junit.jupiter.api.Test;
import java.util.Collections;
import static org.junit.jupiter.api.Assertions.*;
import com.markovai.server.ai.PatchSequenceExtractor;

class DigitMarkovModelTest {

    @Test
    void testBinarizeAndFlatten() {
        int[][] pixels = new int[28][28];
        pixels[0][0] = 200; // should be 1
        pixels[0][1] = 50; // should be 0
        // Patch (0,0) covers pixels (0,0), (0,1), (1,0), (1,1).
        // If (0,0)=1, (0,1)=0, (1,0)=0, (1,1)=0 -> Code = 8*1 + 4*0 + 2*0 + 1*0 = 8.

        int[][] binary = DigitMarkovModel.binarize(pixels, 128);
        assertEquals(1, binary[0][0]);
        assertEquals(0, binary[0][1]);

        PatchSequenceExtractor extractor = new PatchSequenceExtractor();
        int[] seq = extractor.extractSequence(binary);
        // First patch (0,0) -> bit pattern 1000 = 8
        assertEquals(8, seq[0]);
        assertEquals(14 * 14, seq.length);
    }

    @Test
    void testTrainingAndClassification() {
        // Use 16 states for patch model
        DigitMarkovModel model = new DigitMarkovModel(16, new PatchSequenceExtractor());

        // Create a dummy "zero" image (A Box/Square loop)
        // Box from (5,5) to (20,20)
        int[][] pixelsZero = new int[28][28];
        for (int r = 5; r <= 20; r++) {
            pixelsZero[r][5] = 255; // Left
            pixelsZero[r][20] = 255; // Right
        }
        for (int c = 5; c <= 20; c++) {
            pixelsZero[5][c] = 255; // Top
            pixelsZero[20][c] = 255; // Bottom
        }
        DigitImage imgZero = new DigitImage(pixelsZero, 0);

        // Create a dummy "one" image (Vertical line)
        int[][] pixelsOne = new int[28][28];
        for (int r = 5; r <= 20; r++)
            pixelsOne[r][14] = 255; // Vertical line only
        DigitImage imgOne = new DigitImage(pixelsOne, 1);

        model.train(java.util.List.of(imgZero, imgOne));

        // Test classification
        ClassificationResult resZero = model.classifyWithScores(imgZero);
        assertEquals(0, resZero.getPredictedDigit());

        ClassificationResult resOne = model.classifyWithScores(imgOne);
        assertEquals(1, resOne.getPredictedDigit());
    }
}
