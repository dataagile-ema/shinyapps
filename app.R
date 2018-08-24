library(shiny)
library(tidyverse)


houseData <- dplyr::tibble(houseNumber = c(1, 2, 3, 4, 5), squareSpace = c(20,40, 80, 90, 120), price = c(5000, 7000, 10500, 12000, 15000))
n = 5


LearningRate = 0.0001


calcErrorsAndGradients <- function(houseData, weight, bias) {
  
  houseData_Errors_Gradients <- houseData %>%
    mutate(prediction = weight * squareSpace + bias) %>%
    mutate(error = prediction - price) %>%
    mutate(weightGradient = error * squareSpace) %>%
    mutate(gradientBias = error) %>%
    mutate(errrorSquared = error ^ 2)
  
  return (houseData_Errors_Gradients)
}


calcIncrementWeight <- function(houseData_Errors_Gradients, weight) {
  
  increment <- houseData_Errors_Gradients %>%
    summarize(incrementWeight = -sum(houseData_Errors_Gradients$weightGradient) / n * LearningRate)
  
  return(increment$incrementWeight)
}

calcIncrementBias <- function(houseData_Errors_Gradients, bias) {
  
  increment <- houseData_Errors_Gradients %>%
    summarize(incrementBias = -sum(houseData_Errors_Gradients$gradientBias) / n * LearningRate * 1000)
  
  return(increment$incrementBias)
}



houseData_Errors_Gradients <- calcErrorsAndGradients(houseData, 3, 40)

ui <- fluidPage(
  
  sidebarLayout(

      sidebarPanel(
      # --- Update model ----
      tags$div( # headline
        HTML('<h4 style="color:#000;margin-left:0px;">Träna modell</h4>')
      ),
      
      #tags$div( # partition
      #  HTML('<hr style="height:1px; border:none; color:#000; background-color:#000;">')
      #),
      htmlOutput(outputId = "updateModelText"),
      actionButton("train_button", "Uppdatera modell", width = 150)
    ),
    
    
    mainPanel(
      # --- Visualize model training
      htmlOutput(outputId = "modelText"),
      plotOutput(outputId = "plotByHouseNo"), #, width = "100%", height = "200px"),
      htmlOutput(outputId = "roundText"),
      actionButton("reset_button", "Återställ modell", width = 150)
      
    )
    
  )
)



server <- function(input, output, session) {
  
  # --- reactive expressions
  modelData <- reactiveValues(increment = 0, houseData_Errors_Gradients = houseData, weight = 40, bias = 2000) # Defining & initializing the reactiveValues object
  
  incrementWieght <- reactive({
    calcIncrementWeight(modelData$houseData_Errors_Gradients, modelData$weight)
  })
  
  
  incrementBias <- reactive({
    calcIncrementBias(modelData$houseData_Errors_Gradients, modelData$weight)
  })
  

  # --- Visualize model
  output$modelText <- renderUI({
    
    modelData$houseData_Errors_Gradients <- calcErrorsAndGradients(houseData, modelData$weight, modelData$bias)
    text0 <- paste("Modell:")
    text1 <- paste("Pris = k * boyta + m")
    text2 <- paste("k = ", format((modelData$weight), digits = 2), "",'&nbsp;', "m = ", format((modelData$bias), digits = 0))
    HTML("<font face='Courier New' style = font-size:16px>","<font color='#00BFFF'>", "<b>", text0,"</font>", "</b>", text1, "<br>", text2, "<br>", "<font color='#27e833'>", "<b>", "Tränindsdata", "</b>", "</font>")
  })
  

  output$plotByHouseNo <-renderPlot({
    t1 <- ggplot(modelData$houseData_Errors_Gradients, aes(x = houseNumber)) 
    t2 <- t1  + geom_point(color='#00BFFF', aes(y = prediction, size = 30)) 
    t3 <- t2  + geom_point(color = '#27e833', aes(y = price, size = 30)) + xlim(1,5) + ylim(1000, 16000) + theme_bw()
    t4 <- t3 + labs( x = "Hus", y = "Pris") 
    t4 + theme(legend.position = "none", text = element_text(size=16))
  })
  
  output$roundText <- renderUI({
    
    text <- paste("<font face='Courier New' style = font-size:16px>", "Träningsrunda: ", modelData$increment)
    HTML(text)
  })
  
    
  
  #--- Update model ---
  output$updateModelText <- renderUI({
    
    modelData$houseData_Errors_Gradients <- calcErrorsAndGradients(houseData, modelData$weight, modelData$bias)
    
    weightText <- paste("Justera k med:",format(incrementWieght(), digits = 2)) #, "<br>")#, "Nytt värde för k", format(incrementWieght(), digits = 2), "+", format(modelData$weight, digits = 2), "<b>", "=", format(modelData$weight +incrementWieght(), digits = 2), "</b>")
    biasText <- paste("Justera m med:",format(incrementBias(), digits = 2),"<br>")#, "Nytt värde för m", format(incrementBias(), digits = 2), "+", format(modelData$bias, digits = 2), "=", "<b>", format(modelData$bias +incrementBias(), digits = 2), "</b>")
    
    
    HTML("<font face='Courier New' style = font-size:16px>", weightText, "<br>", biasText )
  })
  
  observeEvent(input$train_button, {
    modelData$houseData_Errors_Gradients <- calcErrorsAndGradients(houseData, modelData$weight, modelData$bias)
    modelData$weight = modelData$weight + incrementWieght()
    modelData$bias = modelData$bias + incrementBias()
    modelData$increment = modelData$increment + 1
  })
  
  
  
  observeEvent(input$reset_button, {
    modelData$bias = 2000
    modelData$weight = 40
    modelData$increment = 0
  })
}

# Create a Shiny app object
shinyApp(ui = ui, server = server)

