plugins {
    id("org.springframework.boot")
    id("io.spring.dependency-management")
    java
}

dependencies {
    implementation("org.springframework.boot:spring-boot-starter-web")
    testImplementation("org.springframework.boot:spring-boot-starter-test")
    implementation("org.xerial:sqlite-jdbc:3.45.1.0")
}

tasks.test {
    useJUnitPlatform()
}

val frontendDir = project.rootDir.resolve("client")

val npmInstall by tasks.registering(Exec::class) {
    workingDir = frontendDir
    commandLine("npm", "install")
    inputs.file(frontendDir.resolve("package.json"))
    outputs.dir(frontendDir.resolve("node_modules"))
}

val npmBuild by tasks.registering(Exec::class) {
    dependsOn(npmInstall)
    workingDir = frontendDir
    commandLine("npm", "run", "build")
    inputs.dir(frontendDir.resolve("src"))
    inputs.file(frontendDir.resolve("index.html"))
    inputs.file(frontendDir.resolve("vite.config.js"))
    outputs.dir(frontendDir.resolve("dist")) // Vite output is dist by default, but we configured it to server/static
    // Actually, in vite.config.js we set outDir to '../server/src/main/resources/static'
    // So the output is actually directly into the resource folder.
    // However, Gradle needs to know this task produces resources.
}

tasks.processResources {
    dependsOn(npmBuild)
}

tasks.register<JavaExec>("precompute") {
    group = "application"
    description = "Runs the offline precompute tool"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.markovai.server.tools.DigitDatasetPrecompute")
}
