from pathlib import Path

rule all:
    input:
        Path("reports/plots/knn_spam_accuracy.png"),
        Path("reports/plots/knn_spam_precision.png"),
        Path("reports/plots/knn_spam_recall.png"),
        Path("reports/plots/knn_cancer_accuracy.png"),
        Path("reports/plots/knn_cancer_precision.png"),
        Path("reports/plots/knn_cancer_recall.png"),
        Path("reports/plots/knn_spam_roc_curve.png"),
        Path("reports/plots/knn_cancer_roc_curve.png")
        
        
rule spam_precision_recall:
    input:
        Path("data/spam.csv")
    output:
        Path("reports/plots/knn_spam_recall.png"),
        Path("reports/plots/knn_spam_precision.png"),
        Path("reports/plots/knn_spam_accuracy.png")
    params:
        cli=Path("cli.py")
    shell:
        "python3 {params.cli} precision-recall {input} {output}"
    
rule spam_roc_curve:      
    input:
        Path("data/spam.csv")
    output:
        Path("reports/plots/knn_spam_roc_curve.png")
    params:
        cli=Path("cli.py")
    shell:
        "python3 {params.cli} roc-curve {input} {output}"
    
rule cancer_precision_recall:
    input:
        Path("data/cancer.csv")
    output:
        Path("reports/plots/knn_cancer_recall.png"),
        Path("reports/plots/knn_cancer_precision.png"),
        Path("reports/plots/knn_cancer_accuracy.png")
    params:
        cli=Path("cli.py")
    shell:
        "python3 {params.cli} precision-recall {input} {output}"
    
rule cancer_roc_curve:      
    input:
        Path("data/cancer.csv")
    output:
        Path("reports/plots/knn_cancer_roc_curve.png")
    params:
        cli=Path("cli.py")
    shell:
        "python3 {params.cli} roc-curve {input} {output}"
    
