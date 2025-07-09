```mermaid
graph TD
    %% 1. DEFINE ALL STYLES
    classDef start_node fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#333;
    classDef end_node fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#333;
    classDef process_node fill:#e0f7fa,stroke:#007bff,stroke-width:2px,color:#333;
    classDef decision_node fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#333;
    classDef info_node fill:#f0f0f0,stroke:#6c757d,stroke-width:2px,color:#333;


    %% 2. DEFINE THE GRAPH STRUCTURE
    start((Start)) --> gpu_check{GPU available?};
    
    gpu_check -- No --> cpu_only_options("Use TabPFN Client backend or<br/>Local Version");
    
    gpu_check -- Yes --> task_type{"Type of task?"};

    task_type -- Unsupervised --> unsupervised_type{"What kind of<br/>unsupervised task?"};
    unsupervised_type --> imputation(Imputation);
    unsupervised_type --> data_gen("Data Generation"); 
    unsupervised_type --> density("Density Estimation"); 
    unsupervised_type --> embedding("Get Embeddings"); 

    task_type -- "Prediction Problem" --> text_check{"Contains Text Data?"};
    
    text_check -- Yes --> api_backend("Consider using our API client as<br/>TabPFN backend.<br/>Natively understands text.");
    
    text_check -- No --> ts_check{"Time-Series Data?"};

    ts_check -- Yes --> ts_features["Consider TabPFN-Time-Series features"];
    ts_check -- No --> sample_size_check{"More than 10,000 samples?"};

    ts_features --> sample_size_check;
    api_backend --> sample_size_check;

    sample_size_check -- No --> class_check{"More than 10 classes?"};
    sample_size_check -- Yes --> subsample["TabPFN subsample<br/>samples 10,000"];

    class_check -- No --> rfpfn("RF-PFN");
    class_check -- Yes --> many_class("Many Class");

    subsample --> finetune_check{"Need to Finetune?"};
    rfpfn --> finetune_check;
    many_class --> finetune_check;

    finetune_check -- Yes --> finetuning("Finetuning");
    finetune_check -- No --> interpretability_check{"Need Interpretability?"};

    finetuning --> performance_check{"Performance not<br/>good enough?"};
    interpretability_check -- Yes --> shapley("Shapley Values for TabPFN");
    interpretability_check -- No --> performance_check;
    shapley --> performance_check;

    performance_check -- No --> congrats((Congrats!));
    performance_check -- Yes --> tuning_options("Tuning Options");
    tuning_options --> more_estimators("More estimators on TabPFN");
    tuning_options --> hpo("HPO for TabPFN");
    tuning_options --> post_hoc("Post-Hoc-Ensemble<br/>(AutoTabPFN)");

    %% 3. APPLY STYLES TO NODES
    class Start start_node;
    class Congrats end_node;
    class gpu_check,task_type,unsupervised_type,text_check,ts_check,sample_size_check,class_check,finetune_check,interpretability_check,performance_check decision_node;
    class cpu_only_options,imputation,data_gen,density,embedding,api_backend,ts_features,subsample,rfpfn,many_class,finetuning,shapley,more_estimators,hpo,post_hoc,tuning_options process_node;


    %% 4. ADD CLICKABLE LINKS
    click cpu_only_options "https://github.com/PriorLabs/TabPFN" "TabPFN Backend Options" _blank
    click api_backend "https://github.com/PriorLabs/tabpfn-client" "TabPFN API Client" _blank
    click unsupervised_type "https://github.com/PriorLabs/tabpfn-extensions" "TabPFN Extensions" _blank
    click data_gen "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/unsupervised/generate_data.py" "TabPFN Data Generation Example" _blank
    click density "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/unsupervised/detect_outliers.py" "TabPFN Density Estimation Example" _blank
    click embedding "https://github.com/PriorLabs/tabpfn-extensions/tree/main/examples/embedding" "TabPFN Embedding Example" _blank
    click ts_features "https://github.com/PriorLabs/tabpfn-time-series" "TabPFN Time-Series Example" _blank
    click rfpfn "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/rf_pfn/rf_pfn_example.py" "RF-PFN Example" _blank
    click many_class "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/many_class/many_class_classifier_example.py" "Many Class Example" _blank
    click finetuning "https://github.com/PriorLabs/TabPFN/blob/main/examples/finetune_classifier.py" "Finetuning Example" _blank
    click shapley "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/interpretability/shap_example.py" "Shapley Values Example" _blank
    click post_hoc "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/phe/phe_example.py" "Post-Hoc Ensemble Example" _blank
    click hpo "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/hpo/tuned_tabpfn.py" "HPO Example" _blank
    click subsample "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/large_datasets/large_datasets_example.py" "Large Datasets Example" _blank

    