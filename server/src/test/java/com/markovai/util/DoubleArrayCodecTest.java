package com.markovai.util;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class DoubleArrayCodecTest {

    @Test
    public void testRoundTrip() {
        double[] input = { 1.0, 2.0, 3.14159, -0.5, 0.0 };
        byte[] bytes = DoubleArrayCodec.toBytes(input);
        double[] output = DoubleArrayCodec.fromBytes(bytes);

        Assertions.assertArrayEquals(input, output, 0.0000001);
    }

    @Test
    public void testEmpty() {
        double[] input = {};
        byte[] bytes = DoubleArrayCodec.toBytes(input);
        double[] output = DoubleArrayCodec.fromBytes(bytes);

        Assertions.assertArrayEquals(input, output, 0.0000001);
    }
}
