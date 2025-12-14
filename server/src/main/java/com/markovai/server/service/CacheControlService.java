package com.markovai.server.service;

import com.markovai.db.MarkovChainResultDao;
import org.springframework.stereotype.Service;
import java.io.File;
import java.sql.SQLException;

@Service
public class CacheControlService {

    private static final String DB_PATH;

    static {
        String dataDir = System.getProperty("markov.data.dir", ".");
        DB_PATH = dataDir + File.separator + "markov_cache.db";
    }
    private final MarkovChainResultDao resultDao;

    public CacheControlService() {
        this.resultDao = new MarkovChainResultDao(DB_PATH);
    }

    /**
     * Clears all cached results for a specific chain type and version.
     * Use this when chain parameters or algorithms change.
     */
    public void clearChain(String chainType, String chainVersion) {
        try {
            resultDao.deleteByChain(chainType, chainVersion);
        } catch (SQLException e) {
            throw new RuntimeException("Failed to clear chain cache for " + chainType + "/" + chainVersion, e);
        }
    }
}
