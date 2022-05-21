from utils.molecular_metrics import MolecularMetrics


def all_scores(mols, data, norm=False, reconstruction=False):
    m0 = {k: list(filter(lambda e: e is not None, v)) for k, v in {
        # 'NP score': MolecularMetrics.natural_product_scores(mols, norm=norm),
        'QED score': MolecularMetrics.quantitative_estimation_druglikeness_scores(mols),
        'logP score': MolecularMetrics.water_octanol_partition_coefficient_scores(mols,
                                                                                  norm=norm)}.items()}
    # 'SA score': MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=norm),
    # 'diversity score': MolecularMetrics.diversity_scores(mols, data),
    # 'drugcandidate score': MolecularMetrics.drugcandidate_scores(mols, data)}.items()}

    m1 = {'valid score': MolecularMetrics.valid_total_score(mols) * 100,
          'unique score': MolecularMetrics.unique_total_score(mols) * 100,
          'novel score': MolecularMetrics.novel_total_score(mols, data) * 100}

    return m0, m1
