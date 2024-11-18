classDiagram
    note "1. Inheritance: All GLM types inherit from base GLM class\n2. Abstract Methods: Base class defines interface\n3. Polymorphism: Each GLM type implements same interface\n4. Overriding: Each GLM type provides specific implementations"

    %% Abstract Base Class
    class GLM {
        <<abstract>>
        #Abstract Methods
        +fit()* 
        +predict()*
        #Common Implementation
        +__init__(X, y)
        -_validate_data()
    }

    %% Inheritance & Implementation
    class NormalGLM {
        #Overridden Methods
        +fit() override
        +predict() override
        #Specific Implementation
        -_identity_link(η)
        -_neg_log_likelihood()
    }

    class BernoulliGLM {
        #Overridden Methods
        +fit() override
        +predict() override
        #Specific Implementation
        -_logit_link(η)
        -_neg_log_likelihood()
    }

    class PoissonGLM {
        #Overridden Methods
        +fit() override
        +predict() override
        #Specific Implementation
        -_log_link(η)
        -_neg_log_likelihood()
    }

    %% Show inheritance relationships
    GLM <|-- NormalGLM : Inheritance
    GLM <|-- BernoulliGLM : Inheritance
    GLM <|-- PoissonGLM : Inheritance

    note for GLM "Abstract Base Class\nDefines common interface\nEnforces implementation"
    note for NormalGLM "Polymorphic Implementation\nIdentity Link\nμ = η"
    note for BernoulliGLM "Polymorphic Implementation\nLogit Link\nμ = 1/(1+e^(-η))"
    note for PoissonGLM "Polymorphic Implementation\nLog Link\nμ = exp(η)"



![alt text](image.png)