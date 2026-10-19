PrivEsc-LLM: ACSAC 2026 Artifact

Artifact for "Towards Reliable Local Security Agents: Verifiable Post-Training
for Linux Privilege Escalation" (ACSAC 2026).

Requirements for both claims are in infrastructure/.

Quick check (CPU, about two minutes):
    git clone --branch acsac2026-ae https://github.com/sailab-vienna/privesc-llm.git
    cd privesc-llm
    bash install.sh
    bash claims/claim1/run.sh

Run both claims on a GPU host:
    bash run.sh

Claim 1 recomputes the paper's results from the released evaluation records.
Claim 2 reruns the static-benchmark comparison with three runs per scenario;
it takes 2 to 3 hours, up to a working day including setup.

Each runner ends with PASS, and run.sh ends with PASS once both claims pass.
artifact/ARTIFACT.md maps each paper result to its evidence.

Layout:
    artifact/        The PrivEsc-LLM repository
    infrastructure/  Hardware, software, and access requirements
    claims/          One folder per claim: claim.txt, run.sh, expected/
    install.sh       Sets up the Python environment for Claim 1
    run.sh           Sets up and runs both claims
    license.txt      License
    use.txt          Intended use and limitations
    metadata.toml    Submission metadata
