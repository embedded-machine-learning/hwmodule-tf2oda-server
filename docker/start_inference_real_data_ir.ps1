docker run `
--rm `
-it `
-v C:\Projekte\Docker_training\tf27\eml_projects\test-project:/eml-tools/eml_projects/test-project `
-v C:\Projekte\21_SoC_EML\datasets\dataset-oxford-pets-val-debug:/eml-tools/dataset `
-p 80:80 `
eml_tf2oda_inference:tf2_2.7.0-gpu_2