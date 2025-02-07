module GymnasticMedalsPrediction

import Pkg
using Pkg 
Pkg.add("CSV") 
Pkg.add("DataFrames") 
Pkg.add("CategoricalArrays")
Pkg.add("MLJ")
Pkg.add("DecisionTree")
Pkg.add("ScikitLearn")
Pkg.add("MLJModels")
Pkg.add("MLJBase")
Pkg.status()

using MLJModels
using DecisionTree
using ScikitLearn: @sk_import, fit!, predict
using ScikitLearn.Skcore: train_test_split
using CSV
using DataFrames
using CategoricalArrays
using MLJ
using MLJBase: confusion_matrix, precision, recall, f1score

function prepare_data()
    data = CSV.read("Datasets/Athletes_summer_games.csv", DataFrame)

    # Filtrer pour ne garder que la gymnastique
    databis = filter(row -> row[:Sport] in ["Artistic Gymnastics", "Gymnastics"], data)

    # Garder uniquement les JO entre 2008 et 2024
    filtered_data = filter(row -> row[:Year] >= 2008 && row[:Year] <= 2024, databis)
    dropmissing!(filtered_data)
    filtered_data.Medal = replace(filtered_data.Medal, missing => "No medal")
    filtered_data.Medal = replace(filtered_data.Medal, "Gold" => 1, "Silver" => 2, "Bronze" => 3, "No medal" => 0)

    # Nettoyage des événements
    filtered_data.Event = categorical(filtered_data.Event)
    replace!(filtered_data.Event, "Gymnastics Women's Uneven Bars" => "Barres asymétriques")
    replace!(filtered_data.Event, "Gymnastics Women's Horse Vault" => "Saut Femme")
    replace!(filtered_data.Event, "Gymnastics Women's Floor Exercise" => "Sol Femme")
    replace!(filtered_data.Event, "Gymnastics Women's Balance Beam" => "Poutre Femme")
    replace!(filtered_data.Event, "Gymnastics Men's Rings" => "Anneaux homme")
    replace!(filtered_data.Event, "Gymnastics Men's Floor Exercise" => "Sol homme")
    replace!(filtered_data.Event, "Gymnastics Men's Horse Vault" => "Saut homme")
    replace!(filtered_data.Event, "Gymnastics Men's Parallel Bars" => "Barres parallèles hommes")
    replace!(filtered_data.Event, "Gymnastics Men's Pommelled Horse" => "Cheval d'Arçons")
    replace!(filtered_data.Event, "Gymnastics Men's Horizontal Bar" => "Barre fixe")
    replace!(filtered_data.Event, "Gymnastics Women's Individual All-Around" => "Concours Individuel feminin")
    replace!(filtered_data.Event, "Gymnastics Women's Team All-Around" => "Concours équipes feminin")
    replace!(filtered_data.Event, "Gymnastics Men's Team All-Around" => "Concours équipes masculin")
    replace!(filtered_data.Event, "Gymnastics Men's Individual All-Around" => "Concours Individuel masculin")

    # Colonnes pour chaque type de médailles
    filtered_data.GoldMedals = filtered_data.Medal .== 1
    filtered_data.SilverMedals = filtered_data.Medal .== 2
    filtered_data.BronzeMedals = filtered_data.Medal .== 3
    filtered_data.NoMedals = filtered_data.Medal .== 0

    # Transformer les variables catégorielles
    filtered_data.Sex = categorical(filtered_data.Sex)
    filtered_data.NOC = categorical(filtered_data.NOC)
    filtered_data.Name = categorical(filtered_data.Name)

    # Nettoyage de la variable Age
    filtered_data = dropmissing(filtered_data, :Age)
    filtered_data.Age = Int.(filtered_data.Age)

    # Calculer le nombre de médailles par athlète
    total_Medals = combine(groupby(filtered_data, :Name), [:GoldMedals, :SilverMedals, :BronzeMedals] .=> sum)
    filtered_data = leftjoin(filtered_data, total_Medals, on = :Name)

    return filtered_data
end

function train_model(filtered_data)
    # Fonction d'encodage One-Hot personnalisé
    function one_hot_encode(column, colname_prefix)
        unique_vals = unique(column)
        return DataFrame([Symbol("$(colname_prefix)_$(val)") => (column .== val) for val in unique_vals])
    end

    # Sélectionner les colonnes et encoder
    X = select(filtered_data, [:Age])  # Inclut la colonne `Age` seule pour l'instant
    X = hcat(X, one_hot_encode(filtered_data[!, :Sex], "Sex"))
    X = hcat(X, one_hot_encode(filtered_data[!, :Event], "Event"))
    X = hcat(X, one_hot_encode(filtered_data[!, :NOC], "NOC"))

    y = filtered_data.Medal

    # Diviser les données en ensemble d'entraînement et de test
    train, test = partition(eachindex(y), 0.7)
    X_train, X_test = Matrix(X[train, :]), Matrix(X[test, :])  # Convertir en matrice
    y_train, y_test = y[train], y[test]

    # Configuration et entraînement du modèle de forêt aléatoire
    model = DecisionTreeClassifier(max_depth=2)
    fit!(model, X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = predict(model, X_test)
    
    return model, y_test, y_pred
end

function evaluate_model(y_test, y_pred)
    # Calcul de l'accuracy
    accuracy = sum(y_test .== y_pred) / length(y_test)
    println("Accuracy: ", accuracy)

    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    println("Matrice de confusion : ", cm)

    # Calcul des scores de précision, rappel et F1
    recall_score = recall(y_pred, y_test)
    f1_score = f1score(y_pred, y_test)
    precision_score = 0
    return accuracy, cm, precision_score, recall_score, f1_score
end

filtered_data = prepare_data()

# Entraînement et prédiction du modèle
model, y_test, y_pred = train_model(filtered_data)

# Évaluation du modèle
accuracy, cm, precision_score, recall_score, f1_score = evaluate_model(y_test, y_pred)

# Optionnel : afficher les résultats par événement
prediction = DataFrame(Name = filtered_data.Name[partition(eachindex(filtered_data.Medal), 0.7)[2]], pred_Medal = y_pred, Event = filtered_data.Event[partition(eachindex(filtered_data.Medal), 0.7)[2]])
resultat_par_evnmt = groupby(prediction, :Event)

end