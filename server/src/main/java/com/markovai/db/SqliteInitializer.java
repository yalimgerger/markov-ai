package com.markovai.db;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

public class SqliteInitializer {

    public static void initialize(String dbPath) throws SQLException {
        String url = "jdbc:sqlite:" + dbPath;
        try (Connection conn = DriverManager.getConnection(url)) {
            try (Statement stmt = conn.createStatement()) {
                // Enable WAL mode
                stmt.execute("PRAGMA journal_mode = WAL;");

                // Create digit_image table
                stmt.execute("CREATE TABLE IF NOT EXISTS digit_image (" +
                        "id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                        "image_rel_path TEXT NOT NULL UNIQUE, " +
                        "image_hash TEXT, " +
                        "created_ts INTEGER NOT NULL" +
                        ");");

                // Create markov_chain_result table
                stmt.execute("CREATE TABLE IF NOT EXISTS markov_chain_result (" +
                        "id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                        "image_id INTEGER NOT NULL, " +
                        "chain_type TEXT NOT NULL, " +
                        "chain_version TEXT NOT NULL, " +
                        "scores_blob BLOB NOT NULL, " +
                        "created_ts INTEGER NOT NULL, " +
                        "UNIQUE (image_id, chain_type, chain_version), " +
                        "FOREIGN KEY (image_id) REFERENCES digit_image(id) ON DELETE CASCADE" +
                        ");");

                // Create index
                stmt.execute("CREATE INDEX IF NOT EXISTS idx_chain_lookup " +
                        "ON markov_chain_result (chain_type, chain_version, image_id);");
            }
        }
    }
}
