use std::collections::HashMap;

pub type Feature<T> = (&str, T);
pub type Label = &str;

pub struct NaiveBayes {
    training_docs: Vec<Document>,
    tokenizer: Tokenizer
}

pub struct Model {
    features: Vec<Feature<T>>,
    labels: Vec<Label>,
    label_likelihoods: HashMap<Label, HashMap<Feature<T>, f64>>
}

impl Model {
    pub fn get_likelihoods(&self) -> HashMap<Label, HashMap<Feature<T>, f64>> {
        self.label_likelihoods
    }
}

impl NaiveBayes {
    pub fn train(featureset: Vec<(Vec<Feature<T>>, Label)>) -> Self {
        let model = Model::new();

        let label_totals: HashMap<&str, u32> = HashMap::new();

        let feature_counts: HashMap<Label, HashMap<Feature<T>, u32>> = HashMap::new();
        for f in featureset.iter() {
            let label = f.1;
            let label_totals_incr = label_totals.entry(*label).or_insert(0);
            *label_totals_incr += 1;
            let label_entry = feature_counts.entry(*label).or_insert(HashMap::new());
            let feature_count_entry = *label_entry.entry(f).or_insert(0);
            *feature_count_entry += 1;
        }

        // normalize counts to get likelihoods
        for fs in feature_counts.iter() {
            model.label_likelihoods.insert(fs.0, HashMap::new());
            for f in fs.iter() {
                model.label_likelihoods[fs.0].insert(f.0, f.1 / label_totals[fs.0]);
            }
        }

        model
    }
    pub fn test(featureset: Vec<(Vec<Feature<T>>, Label)>) -> Self {
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn train_test() {
        let mut nb = NaiveBayes::new();

        let featureset: Vec<(Vec<Feature<&str>>, Label)> = vec![
            (vec![("last-letter", 'a')], "female"),
            (vec![( "last-letter", 'e')], "female"),
            (vec![( "last-letter", 'k')], "male"),
            (vec![( "last-letter", 'p')], "male"),
        ];

        let test_label_likelihoods = HashMap::new();
        test_label_likelihoods.insert("male", ({ "last-letter": 'k'}, 0.9));

        let model = nb.train(&featureset);

        assert!(&model.get_label_likelihoods() == test_label_likelihoods);
    }
}