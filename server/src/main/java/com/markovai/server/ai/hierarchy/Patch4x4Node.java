package com.markovai.server.ai.hierarchy;

import com.markovai.server.ai.DigitImage;
import com.markovai.server.ai.DigitMarkovModel;
import com.markovai.server.ai.DigitPatch4x4UnigramModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import java.util.ArrayList;

public class Patch4x4Node implements DigitFactorNode {
    private static final Logger logger = LoggerFactory.getLogger(Patch4x4Node.class);

    private final String id;
    private final DigitPatch4x4UnigramModel model;
    private final double smoothingLambda;

    public Patch4x4Node(String id, DigitPatch4x4UnigramModel model, double smoothingLambda) {
        this.id = id;
        this.model = model;
        this.smoothingLambda = smoothingLambda;
    }

    @Override
    public String getId() {
        return id;
    }

    @Override
    public List<DigitFactorNode> getChildren() {
        return Collections.emptyList();
    }

    @Override
    public NodeResult computeForImage(DigitImage img, Map<String, NodeResult> childResults) {
        // Reuse existing binarization
        int[][] binary = DigitMarkovModel.binarize(img.pixels, 128);

        // 1. Extract Sequence
        List<Integer> patchSymbols = new ArrayList<>();
        for (int r = 0; r < 7; r++) {
            for (int c = 0; c < 7; c++) {
                patchSymbols.add(DigitPatch4x4UnigramModel.encodePatch(binary, r * 4, c * 4));
            }
        }
        int nSteps = patchSymbols.size(); // Should be 49

        // 2. Compute Smoothed Sum
        double[] smoothedSum = new double[10];

        for (int d = 0; d < 10; d++) {
            double prevLp = 0.0;
            boolean first = true;
            double sum = 0.0;

            for (Integer symbol : patchSymbols) {
                double lp = model.logProbForSymbol(d, symbol);

                if (first) {
                    sum += lp;
                    first = false;
                } else {
                    double diff = lp - prevLp;
                    sum += lp - smoothingLambda * Math.abs(diff);
                }
                prevLp = lp;
            }
            smoothedSum[d] = sum;
        }

        // 3. Convert to Average
        double[] avgLogL = new double[10];
        for (int d = 0; d < 10; d++) {
            avgLogL[d] = smoothedSum[d] / nSteps;
        }

        if (logger.isDebugEnabled()) {
            double minAvg = Double.MAX_VALUE;
            double maxAvg = Double.MIN_VALUE;
            for (double val : avgLogL) {
                if (val < minAvg)
                    minAvg = val;
                if (val > maxAvg)
                    maxAvg = val;
            }
            logger.debug("Patch4x4Node {} smoothingLambda={} [Label {}]: Steps={}. AvgLogL range=[{:.4f}, {:.4f}]",
                    id, smoothingLambda, img.label, nSteps, minAvg, maxAvg);
        }

        return new NodeResult(avgLogL);
    }
}
