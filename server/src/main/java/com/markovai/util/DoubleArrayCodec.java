package com.markovai.util;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;

public class DoubleArrayCodec {

    public static byte[] toBytes(double[] scores) {
        if (scores == null) {
            return null;
        }
        ByteBuffer buffer = ByteBuffer.allocate(scores.length * Double.BYTES);
        buffer.asDoubleBuffer().put(scores);
        return buffer.array();
    }

    public static double[] fromBytes(byte[] bytes) {
        if (bytes == null) {
            return null;
        }
        DoubleBuffer buffer = ByteBuffer.wrap(bytes).asDoubleBuffer();
        double[] scores = new double[buffer.remaining()];
        buffer.get(scores);
        return scores;
    }
}
