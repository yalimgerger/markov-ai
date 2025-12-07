package com.markovai.server.ai.hierarchy;

import com.markovai.server.ai.DigitImage;
import java.util.List;
import java.util.Map;

public interface DigitFactorNode {

    // A unique id for this node, used as a key in maps and config later.
    String getId();

    // Children of this node in the factor graph or hierarchy.
    // Leaf nodes (like row, column, patch) will return an empty list.
    List<DigitFactorNode> getChildren();

    // Compute this node's NodeResult for the given image, using any already
    // computed child results.
    // For now, we are only defining the signature. Leaf nodes will ignore
    // childResults.
    NodeResult computeForImage(DigitImage img, Map<String, NodeResult> childResults);
}
