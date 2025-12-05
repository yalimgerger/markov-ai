import { defineConfig } from 'vite'

export default defineConfig({
    build: {
        outDir: '../server/src/main/resources/static',
        emptyOutDir: true
    }
})
