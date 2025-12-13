
tasks.register<JavaExec>("fastVerify") {
    group = "verification"
    dependsOn(tasks.compileJava)
    classpath = files(sourceSets.main.get().output.classesDirs) + files("src/main/resources") + configurations["runtimeClasspath"]
    mainClass.set("com.markovai.server.MarkovAiApplication")
    args = listOf("--verifyFeedbackSweep=true", "--printSweepAsCSV=true")
    // Increase heap
    maxHeapSize = "2g"
}
