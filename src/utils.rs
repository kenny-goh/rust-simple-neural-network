pub struct Utils {}
use std::collections::HashMap;
use std::fs::File;
use std::io::{Write, Error, Read};
use ndarray::Array2;

impl Utils {
    pub fn serialize(params: &HashMap<String, Array2<f64>>, filename: &str) ->Result<(), Error> {
        let serialized_json = serde_json::to_string(&params).unwrap();
        let path =  filename;
        let mut output = File::create(path)?;
        write!(output, "{}", serialized_json)?;
        Ok(())
    }

    pub fn deserialize(filename: &str)->Result<HashMap<String,Array2<f64>>, Error>  {
        let mut file = File::open(filename)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let params: HashMap<String,Array2<f64>> = serde_json::from_str(&contents)?;
        Ok(params)
    }
}

