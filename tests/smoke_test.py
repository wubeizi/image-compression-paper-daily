from paper_daily import TOPICS, score_relevance

CASES = [
    (
        "Will My Assistant Remember My Allergy? What Personal LLM Assistants Forget When Conversation Memory Is Compressed",
        "This paper studies compressed conversational memory in personal LLM assistants.",
        False,
    ),
    (
        "Investigation on Flexural and Compression Strength and Microstructures of Vacuum Assisted 3D Printed Samples",
        "Compression strength and microstructures of printed samples are studied.",
        False,
    ),
    (
        "Towards Practical Compression of 3D Gaussian Splatting",
        "We compress 3D Gaussian splatting representations for efficient rendering.",
        False,
    ),
    (
        "VQ-LIC: Shared Vector-Quantized Learned Image Compression on a Resource-Constrained FPGA",
        "Learned image compression using vector quantization and codebooks. Rate-distortion results are reported in BPP, PSNR and MS-SSIM.",
        True,
    ),
    (
        "ProGIC: Progressive and Lightweight Generative Image Compression with Residual Vector Quantization",
        "We propose generative image compression with residual vector quantization, progressive bitstreams and perceptual quality evaluation.",
        True,
    ),
    (
        "Generative Image Compression by Estimating Gradients of the Rate-variable Feature Distribution",
        "Generative image compression uses diffusion modeling to reconstruct photorealistic images at low bitrate.",
        True,
    ),
]

for title, abstract, expected in CASES:
    matched = []
    paper = {"title": title, "abstract": abstract, "authors": []}
    for topic_name, topic in TOPICS.items():
        score, _ = score_relevance(paper, topic)
        if score >= 11:
            matched.append(topic_name)
    ok = bool(matched) == expected
    print(f"{'PASS' if ok else 'FAIL'} | {title} | {matched}")
    if not ok:
        raise SystemExit(1)
