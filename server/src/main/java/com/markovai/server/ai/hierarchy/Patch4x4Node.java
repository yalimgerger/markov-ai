package com.markovai.server.ai.hierarchy;

import com.markovai.server.ai.DigitImage;
import com.markovai.server.ai.DigitMarkovModel;
import com.markovai.server.ai.DigitPatch4x4UnigramModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class Patch4x4Node implements DigitFactorNode {
    private static final Logger logger = LoggerFactory.getLogger(Patch4x4Node.class);

    private final String id;
    private final DigitPatch4x4UnigramModel model;

    public Patch4x4Node(String id, DigitPatch4x4UnigramModel model) {
        this.id = id;
        this.model = model;
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

        double[] sumLogL = model.sumLogLikelihoodsForImage(binary);
        int nSteps = model.getNumPatchesPerImage(); // 49

        double[] avgLogL = new double[10];
        for (int d = 0; d < 10; d++) {
            avgLogL[d] = sumLogL[d] / nSteps;
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
            logger.debug("Patch4x4Node {} [Label {}]: Steps={}. AvgLogL range=[{:.4f}, {:.4f}]",
                    id, img.label, nSteps, minAvg, maxAvg);
        }

        return new NodeResult(avgLogL);
    }
}
