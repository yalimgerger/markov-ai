package com.markovai.server.ai;

import org.junit.jupiter.api.Test;
import java.util.Collections;
import static org.junit.jupiter.api.Assertions.*;

class DigitMarkovModelTest {

    @Test
    void testBinarizeAndFlatten() {
        int[][] pixels = new int[28][28];
        pixels[0][0] = 200; // should be 1
        pixels[0][1] = 50; // should be 0

        int[][] binary = DigitMarkovModel.binarize(pixels, 128);
        assertEquals(1, binary[0][0]);
        assertEquals(0, binary[0][1]);

        int[] flat = DigitMarkovModel.flattenBinaryImage(binary);
        assertEquals(1, flat[0]);
        assertEquals(0, flat[1]);
        assertEquals(28 * 28, flat.length);
    }

    @Test
    void testTrainingAndClassification() {
        DigitMarkovModel model = new DigitMarkovModel();

        // Create a dummy "zero" image (mostly 0s, some 1s pattern)
        int[][] pixelsZero = new int[28][28];
        for (int i = 5; i < 20; i++)
            pixelsZero[i][5] = 255; // Vertical line
        DigitImage imgZero = new DigitImage(pixelsZero, 0);

        // Create a dummy "one" image (different pattern)
        int[][] pixelsOne = new int[28][28];
        for (int i = 5; i < 20; i++)
            pixelsOne[i][14] = 255; // Vertical line in middle
        DigitImage imgOne = new DigitImage(pixelsOne, 1);

        model.train(java.util.List.of(imgZero, imgOne));

        // Test classification
        ClassificationResult resZero = model.classifyWithScores(imgZero);
        assertEquals(0, resZero.getPredictedDigit());

        ClassificationResult resOne = model.classifyWithScores(imgOne);
        assertEquals(1, resOne.getPredictedDigit());
    }
}
