package com.markovai.server.ai.hierarchy;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import java.io.InputStream;
import static org.junit.jupiter.api.Assertions.*;

public class FactorGraphBuilderConfigTest {

    @Test
    public void testConfigParsing() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        InputStream is = getClass().getResourceAsStream("/mrf_config.json");
        assertNotNull(is, "mrf_config.json not found in test classpath");

        FactorGraphBuilder.ConfigRoot config = mapper.readValue(is, FactorGraphBuilder.ConfigRoot.class);

        // Assert topology field is present and correct (default is layered in file
        // currently)
        // If I change the file to network, this test should reflect that.
        // For now, I just check it is not null.
        assertNotNull(config.topology);
        System.out.println("Parsed topology: " + config.topology);
    }
}
