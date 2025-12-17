package com.markovai.server.util;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.InputStream;
import java.io.File;

public class DataPathResolver {

    public static String resolveDataDirectory() {
        // 1. Check System Property
        String sysProp = System.getProperty("markov.data.dir");
        if (sysProp != null && !sysProp.isEmpty()) {
            return sysProp;
        }

        // 2. Check Config File
        try {
            ObjectMapper mapper = new ObjectMapper();
            try (InputStream is = DataPathResolver.class.getResourceAsStream("/mrf_config.json")) {
                if (is != null) {
                    JsonNode root = mapper.readTree(is);
                    if (root.has("markov_data_directory")) {
                        String configDir = root.get("markov_data_directory").asText();
                        if (configDir != null && !configDir.isEmpty()) {
                            return configDir;
                        }
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Failed to read markov_data_directory from config: " + e.getMessage());
        }

        // 3. Default
        return ".";
    }

    public static String resolveDbPath() {
        return resolveDataDirectory() + File.separator + "markov_cache.db";
    }
}
