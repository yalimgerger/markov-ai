package com.markovai.server.service;

import com.markovai.server.ai.Patch4x4FeedbackConfig;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FeedbackConfigLoadingTest {

    @Test
    public void testLoadConfigFromDefaultFile() {
        MarkovTrainingService service = new MarkovTrainingService();

        // Test "row" config loading from mrf_config.json (which should be present)
        Patch4x4FeedbackConfig rowConfig = service.getFeedbackConfigOrDefault("row");
        assertNotNull(rowConfig, "Row config should not be null");
        // In mrf_config.json, decayRate is 1.0e-4 and applyDecayEveryNUpdates is 5000
        assertEquals(5000, rowConfig.applyDecayEveryNUpdates);
        assertEquals(1.0e-4, rowConfig.decayRate, 1e-9);

        // Test "col" config
        Patch4x4FeedbackConfig colConfig = service.getFeedbackConfigOrDefault("col");
        assertNotNull(colConfig, "Col config should not be null");
        assertEquals(5000, colConfig.applyDecayEveryNUpdates);

        // Test "patch4x4" config
        Patch4x4FeedbackConfig patchConfig = service.getFeedbackConfigOrDefault("patch4x4");
        assertNotNull(patchConfig, "Patch config should not be null");
        // In mrf_config.json for patch4x4, applyDecayEveryNUpdates is 0
        assertEquals(0, patchConfig.applyDecayEveryNUpdates);
    }

    @Test
    public void testDefaultsForMissingNode() {
        MarkovTrainingService service = new MarkovTrainingService();

        // Request a non-existent node ID to trigger default fallback
        Patch4x4FeedbackConfig missingConfig = service.getFeedbackConfigOrDefault("nonexistent_node");
        assertNotNull(missingConfig);

        // Should match the hardcoded Row/Col default fallback
        // adjScale = 0.10, eta = 0.003
        assertEquals(0.10, missingConfig.adjScale, 1e-9);
        assertEquals(0.003, missingConfig.eta, 1e-9);
        assertTrue(missingConfig.frequencyScalingEnabled);
        assertEquals("GLOBAL_SQRT", missingConfig.frequencyScalingMode);
        // Default fallback has applyDecayEveryNUpdates=0 (from disabled() + custom sets
        // which didn't touch it)
        assertEquals(0, missingConfig.applyDecayEveryNUpdates);
    }
}
