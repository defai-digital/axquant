# Security policy

## Reporting a vulnerability

Please use [GitHub private vulnerability reporting](https://github.com/defai-digital/axquant/security/advisories/new)
for suspected vulnerabilities. Do not include credentials, private checkpoint data, or working
exploits in a public issue.

## Trust model

AXQuant treats checkpoint contents, calibration inputs, benchmark inputs, and model-generated code
as untrusted data. Artifact checksums, strict schemas, and conversion coverage checks protect the
quantization and release evidence chain; they are not a general-purpose host isolation boundary.

Executable coding-suite tasks are the highest-risk surface. On macOS they run through the
system-provided `/usr/bin/sandbox-exec` with a deny-default Seatbelt profile. The scorer:

- permits reads only from Apple's `system.sb` runtime paths, the sealed task-input tree, the task
  output tree, and explicit toolchain roots;
- permits writes only in the task output tree;
- denies network operations;
- denies process creation while generated code and its tests execute;
- applies CPU, wall-clock, memory, process, file-size, file-descriptor, and output limits; and
- requires completion evidence emitted after the hidden tests, so a clean early exit is not a pass.

Compiler processes may create subprocesses because Rust, Go, and TypeScript toolchains require
them. Their filesystem and network restrictions remain active, and generated programs are not run
during the compile phase.

The sandbox policy digest in coding-suite evidence binds the policy contract and renderer version.
The exact toolchain executable paths and versions are bound separately in the suite manifest.
Formal evidence accepts only Apple's SIP-protected `/usr/bin/sandbox-exec`; an arbitrary compatible
wrapper is rejected.

## Limitations

Seatbelt is a same-kernel sandbox, and Apple marks `sandbox-exec` as deprecated. It does not defend
against kernel vulnerabilities, compromised toolchains, or an attacker who already controls the
evaluation account. Toolchain roots must be readable for dynamic libraries and standard packages;
do not place credentials in those roots.

Completion evidence prevents accidental or naive early-success termination. It is not a
cryptographic anti-cheating boundary against code that can inspect and deliberately manipulate its
own language runtime. Certification should evaluate ordinary model output, not adversarial programs
designed with prior knowledge of the private scorer implementation.

Network denial does not make archived output automatically safe to publish. Raw model output and
scorer logs can contain sensitive text supplied in prompts or fixtures and must remain private until
the publication privacy checks pass.

## Safe operation

For coding evaluation of third-party or otherwise adversarial checkpoints:

1. Use a dedicated macOS account or disposable evaluation host with no valuable credentials.
2. Unmount unrelated removable, network, and archive volumes.
3. Keep the sandbox work root on disposable storage and never point it at a source or credential
   directory.
4. Verify the frozen toolchain manifest and run `verify-coding-suite` before formal evaluation.
5. Inspect and privacy-scan raw evidence before copying or publishing it.

Inspection, planning, and conversion do not execute coding-suite model output. Prefer those paths
when executable evaluation is unnecessary.
