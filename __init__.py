"""
Coexpression Analysis Toolkit
==============================

Gene expression correlation and coexpression analysis tools for single-cell transcriptomics.

Main Classes
------------
GeneSetCorrTester : Main analysis class
    Compute correlations, perform statistical tests, and visualize results.

Features
--------
- Pearson correlation analysis between genes
- Gene set enrichment testing (t-test, Wilcoxon, permutation)
- Group-wise correlation analysis
- Comprehensive visualization tools
- Support for different data layers (raw counts, normalized, etc.)

Usage Example
-------------
>>> from Coexpression import GeneSetCorrTester
>>> 
>>> # Initialize analyzer
>>> tester = GeneSetCorrTester(layer='counts', groupby='celltype')
>>> 
>>> # Fit to data and target gene
>>> tester.fit(adata, target_gene='IL17RD')
>>> 
>>> # Compute all correlations
>>> results = tester.compute_all_correlations()
>>> 
>>> # Test gene set significance
>>> gene_set = ['GENE1', 'GENE2', 'GENE3']
>>> stats = tester.test_gene_set_significance(gene_set)
>>> 
>>> # Visualize results
>>> tester.plot_top_overall(topk=20)
>>> tester.plot_compare_distributions(gene_set)

Last Updated: 25.11.02 by GEB
Version: 2.1.0
"""

from .Coexpression import GeneSetCorrTester

__version__ = "2.1.0"
__author__ = "GEB"
__email__ = "gaeunbyeon1102@gmail.com"
__all__ = ["GeneSetCorrTester"]

# Dependency check
def _check_dependencies():
    """Check if required packages are installed."""
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'scipy': 'scipy',
        'matplotlib': 'matplotlib.pyplot',
        'statsmodels': 'statsmodels.stats.multitest'
    }
    
    missing = []
    for name, module in required.items():
        try:
            __import__(module.split('.')[0])
        except ImportError:
            missing.append(name)
    
    if missing:
        raise ImportError(
            f"Missing required dependencies: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}"
        )

# Run dependency check on import
try:
    _check_dependencies()
except ImportError as e:
    import warnings
    warnings.warn(str(e), ImportWarning)
