module GymnasticMedalsPrediction

# Importation des packages nécessaires
using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("CategoricalArrays")
Pkg.add("MLJ")
Pkg.add("MLJModelInterface")
Pkg.add("DecisionTree")
Pkg.add("ScikitLearn")
Pkg.add("MLJModels")
Pkg.add("MLJBase")
Pkg.status()

using MLJModels
using DecisionTree
using MLJModelInterface: transform  # Explicitly import transform
using ScikitLearn: @sk_import, fit!, predict_proba
using ScikitLearn.Skcore: train_test_split
using CSV
using DataFrames
using CategoricalArrays
using MLJ
using MLJBase: confusion_matrix, precision, recall, f1score

# Fonction pour préparer les données
function prepare_data()
    data = CSV.read("Datasets/Athletes_summer_games.csv", DataFrame)

    # Filtrer pour la gymnastique uniquement
    databis = filter(row -> row[:Sport] in ["Artistic Gymnastics","Gymnastics"], data)

    # Filtrer pour les 5 derniers JO
    filtered_data = filter(row -> row[:Year] >= 2008 && row[:Year] <= 2024, databis)
    dropmissing!(filtered_data)
    filtered_data.Medal=replace(filtered_data.Medal, missing =>"No medal")
    filtered_data.Medal=replace(filtered_data.Medal, "Gold" =>1, "Silver" => 2, "Bronze" => 3, "No medal" => 0)

    # Nettoyage des événements
    filtered_data.Event=categorical(filtered_data.Event)
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

    # Création de colonnes pour chaque type de médaille
    filtered_data.GoldMedals = filtered_data.Medal .== 1
    filtered_data.SilverMedals = filtered_data.Medal .== 2
    filtered_data.BronzeMedals = filtered_data.Medal .== 3
    filtered_data.NoMedals = filtered_data.Medal .== 0
    
    # Transformation des variables en catégorielles
    filtered_data.Sex = categorical(filtered_data.Sex)
    filtered_data.NOC = categorical(filtered_data.NOC)
    filtered_data.Name = categorical(filtered_data.Name)

    filtered_data = dropmissing(filtered_data, :Age)
    filtered_data.Age = Int.(filtered_data.Age)

    # Nombre total de médailles par athlète
    total_Medals = combine(groupby(filtered_data, :Name), [:GoldMedals, :SilverMedals, :BronzeMedals] .=> sum)
    filtered_data = leftjoin(filtered_data, total_Medals, on = :Name)

    return filtered_data
end

# Fonction pour entraîner un modèle pour une épreuve spécifique
function train_model_for_event(filtered_data, event_name::String)
    # Filter for the specific event
    event_data = filter(row -> row[:Event] == event_name, filtered_data)
    if nrow(event_data) == 0
        error("Aucune donnée trouvée pour l'épreuve: $event_name")
    end

    # Select features and target variable
    X = select(event_data, [:Sex, :NOC, :Age])
    y = coerce(event_data.Medal, Multiclass)

    # OneHot Encoding using MLJ's OneHotEncoder
    encoder = OneHotEncoder()  # from MLJ
    mach_enc = machine(encoder, X)
    fit!(mach_enc)
    X_enc = transform(mach_enc, X)  # Transform the data using the fitted machine
    X_enc = DataFrame(X_enc)

    # Split into training and testing sets
    train, test = partition(eachindex(y), 0.7)
    X_train, X_test = X_enc[train, :], X_enc[test, :]
    y_train, y_test = y[train], y[test]

    # Load and train RandomForest model
    forest = RandomForestClassifier(n_trees = 100, max_depth = 5)
    mach = machine(forest, X_train, y_train) |> fit!

    # Probability predictions
    y_pred_proba = predict(mach, X_test, proba=true)

    return mach, X_test, y_test, y_pred_proba
end

# Fonction pour évaluer le modèle
function evaluate_model(y_test, y_pred_proba)
    # Conversion des probabilités en prédictions finales
    y_pred = argmax.(y_pred_proba) .- 1  # Ajustement pour que les labels correspondent à 0-3
    
    # Calcul de la précision
    accuracy = mean(y_pred .== y_test)
    println("Précision: ", accuracy)

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    println("Matrice de confusion:\n", cm)

    # Calcul des scores de précision, rappel et F1
    precision_score = precision(y_pred, y_test)
    recall_score = recall(y_pred, y_test)
    f1_score = f1score(y_pred, y_test)

    return accuracy, cm, precision_score, recall_score, f1_score
end

# Fonction pour prédire les probabilités de médailles pour chaque athlète d'une épreuve
function predict_medal_probabilities(mach, event_data)
    # Encodage des données de test
    X_event = select(event_data, [:Sex, :NOC, :Age])
    encoder = OneHotEncoder()
    fit!(encoder, X_event)
    X_enc = transform(encoder, X_event)
    X_enc = DataFrame(X_enc)

    # Prédictions de probabilité
    y_pred_proba = predict_proba(mach, X_enc)
    predictions = DataFrame(Name = event_data.Name, Predicted_Prob_Medal = y_pred_proba, Event = event_data.Event)
    
    return predictions
end

# Exemple d'utilisation
filtered_data = prepare_data()
mach, X_test, y_test, y_pred_proba = train_model_for_event(filtered_data, "Concours Individuel feminin")
accuracy, cm, precision_score, recall_score, f1_score = evaluate_model(y_test, y_pred_proba)
predictions = predict_medal_probabilities(mach, filtered_data[filtered_data.Event .== "Concours Individuel feminin", :])

end