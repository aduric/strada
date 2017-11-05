use std::collections::HashMap;
use std::hash::Hash;

pub type Feature<'a, T> = (&'a str, T);
pub type Label<'a> = &'a str;

pub struct NaiveBayes {}

pub struct Model<'a, T> {
    features: Vec<Feature<'a, T>>,
    labels: Vec<Label<'a>>,
    label_likelihoods: HashMap<Label<'a>, HashMap<Feature<'a, T>, f64>>
}

impl<'a, T> Model<'a, T> {
    pub fn new() -> Self {
        Model {
            features: Vec::new(),
            labels: Vec::new(),
            label_likelihoods: HashMap::new()
        }
    }
    pub fn get_likelihoods(self) -> HashMap<Label, HashMap<Feature<T>, f64>> {
        self.label_likelihoods
    }
}

impl NaiveBayes {
    pub fn train<'a, 'b, T>(featureset: Vec<(Feature<'a, T>, Label<'b>)>) -> Model<'a, T>
        where T: Eq + Hash {

        let model: Model<'a, T> = Model::new();

        let label_totals: HashMap<Label, u32> = HashMap::new();
        let feature_counts: HashMap<Label, HashMap<Feature<T>, u32>> = HashMap::new();

        for f in featureset {
            let feature = f.0;
            let label = f.1;
            let label_totals_incr = label_totals.entry(label).or_insert(0);
            *label_totals_incr += 1;
            let label_entry = feature_counts.entry(label).or_insert(HashMap::new());
            let feature_count_entry = label_entry.entry(feature).or_insert(0);
            *feature_count_entry += 1;
        }

        // normalize counts to get likelihoods
        for fs in feature_counts {
            let l = String::from(fs.0);
            model.label_likelihoods.insert(&l, HashMap::new());
            for fsv in fs.1 {
                model.label_likelihoods[fs.0].insert(fsv.0, fsv.1 as f64 / label_totals[fs.0] as f64);
            }
        }

        model
    }
    pub fn test<'a, T>(featureset: Vec<Feature<'a, T>>, model: Model<T>) -> Label<'a> {
        "test"
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn train_test() {
        let mut nb = NaiveBayes::new();

        let featureset: Vec<(Feature<char>, Label)> = vec![
            (("last-letter", 'a'), "female"),
            (( "last-letter", 'e'), "female"),
            (( "last-letter", 'k'), "male"),
            (( "last-letter", 'p'), "male"),
        ];

        let test_label_likelihoods: HashMap<Label, (Feature<char>, f64)> = HashMap::new();
        test_label_likelihoods.insert("male", (( "last-letter", 'k'), 0.9));

        let model = nb.train(&featureset);

        assert!(&model.get_label_likelihoods() == test_label_likelihoods);
    }
}