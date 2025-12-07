package com.markovai.server.ai;

/**
 * Represents a digit image from the MNIST dataset.
 */
public class DigitImage {
    // pixels[row][col] is a grayscale value in [0, 255]
    public int[][] pixels; // size 28 x 28
    public int label; // 0 to 9

    public DigitImage(int[][] pixels, int label) {
        this.pixels = pixels;
        this.label = label;
    }
}
