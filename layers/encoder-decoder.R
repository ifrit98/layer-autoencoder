source_python("utils/mnist-keras-data.py")

# TODO: What you really might want is a model?  
# KerasWrapper only modifies a single input layer


keras_model_simple_mlp <- function(num_classes, 
                                   use_bn = FALSE, use_dp = FALSE, 
                                   name = NULL) {
  
  # define and return a custom model
  keras_model_custom(name = name, function(self) {
    
    # create layers we'll need for the call (this code executes once)
    self$dense1 <- layer_dense(units = 32, activation = "relu")
    self$dense2 <- layer_dense(units = num_classes, activation = "softmax")
    if (use_dp)
      self$dp <- layer_dropout(rate = 0.5)
    if (use_bn)
      self$bn <- layer_batch_normalization(axis = -1)
    
    # implement call (this code executes during training & inference)
    function(inputs, mask = NULL) {
      x <- self$dense1(inputs)
      if (use_dp)
        x <- self$dp(x)
      if (use_bn)
        x <- self$bn(x)
      self$dense2(x)
    }
  })
}


# TODO: 
#' How deep do you want en(de)coder?
#' What are the params/args to each layer
#' 
CoderWrapper <- 
  R6::R6Class("CoderWrapper",

    inherit = KerasWrapper,
    
    public = list(
      regularizer = NULL,
      
      initialize = function(regularizer) {
        self$regularizer = regularizer
        browser()
      },
      
      build = function(input_shape) {
        super$build(input_shape)
        
        browser()
        
        self$layers <- purrr::map2(
          .x = rep('encoder', num_layers),
          .y = if (is_lst)
            hidden_dim
          else
            rep(hidden_dim, num_layers),
          .f = map2_fn
        )
      },
      
      call = function(x, mask = NULL) {}
      
      add_loss = function(losses, inputs = NULL) {},

      add_weight = function(name,
                            shape,
                            dtype = NUL,
                            initializer = NULL,
                            regularizer = NULL,
                            trainable = TRUE,
                            constraint = NULL) {}
    )
)


layer_coder_wrapper <- 
  function(object, layer, regularizer = NULL) {
    create_wrapper(
      CoderWrapper,
      object,
      list(layer = layer,
           regularizer = regularizer)
    )
  }


EncoderDecoder <-
  R6::R6Class("EncoderDecoder",

    inherit = KerasLayer,

    public = list(
      hidden_dim = NULL,
      mode = NULL,
      original_dim = NULL,
      hidden_layer = NULL,
      output_layer = NULL,

      initialize = function(mode, hidden_dim, original_dim) {
        self$mode <- mode
        self$hidden_dim <- hidden_dim
        self$original_dim <- original_dim # Infer from input_shape?
      },

      build = function(input_shape) {
        # TODO: replace with tensorflow empty function
        if (rlang::is_empty(self$original_dim))
          self$original_dim <- input_shape[[length(input_shape)]]

        self$hidden_layer <- layer_dense(
          units = self$hidden_dim,
          activation = 'relu',
          kernel_initializer = 'he_uniform'
        )

        self$output_layer <- layer_dense(
          units =
            if (self$mode == 'decoder')
              self$original_dim else self$hidden_dim,
          activation = 'relu',
          kernel_initializer = 'he_uniform'
        )
      },

      call = function(x, mask = NULL) {

        activation <- self$hidden_layer(x)

        output <- self$output_layer(activation)

        output
      },

      compute_output_shape = function(input_shape) {
        input_shape
      }

    )
  )



EncoderDecoderV2 <- 
  R6::R6Class("EncoderDecoderV2",
              
    inherit = KerasLayer,
    
    public = list(
      mode = NULL,
      num_layers = NULL,
      hidden_dims = NULL,
      original_dim = NULL,
      code_dim = NULL,
      activation = NULL,
      hidden_layers = NULL,
      
      initialize = function(mode,
                            num_layers,
                            hidden_dims,
                            original_dim,
                            code_dim,
                            activation) {
        self$mode         <- mode
        self$num_layers   <- num_layers
        self$hidden_dims  <- hidden_dims 
        self$original_dim <- original_dim
        self$code_dim     <- code_dim
        self$activation   <- activation
      },
      
      build = function(input_shape) {
        
        if(is_empty(self$original_dim) & self$mode == "decoder")
          stop("Original dimension must be supplied if mode == \"decoder\".") 
        
        if(self$mode == "decoder" & is_empty(self$code_dim))
          stop("Code dim must be supplied when mode == \"decoder\".")
        
        if(is_empty(self$original_dim))
          self$original_dim <- input_shape[[length(input_shape)]]
        
        if (self$mode == "decoder")
          self$hidden_dims  <- c(self$hidden_dims, self$original_dim)  

        input_dims <- dplyr::lag(self$hidden_dims)
        
        input_dims[[1]] <- 
          if (self$mode == "encoder") self$original_dim else self$code_dim
        
        get_layer <- function(in_features, out_features) {

            W <- self$add_weight(
              name = "W_",
              shape = list(in_features, out_features),
              initializer = initializer_he_normal())
            
            b <- self$add_weight(
              name = "b_",
              shape = list(out_features),
              initializer = initializer_zeros())
            
            c(W, b)
          }
        
        self$hidden_layers <- 
          map2(input_dims, self$hidden_dims, get_layer)
          
      },
      
      call = function(x, mask = NULL) {
        
        out <- x
        
        for (layer in self$hidden_layers) {
          W   <- layer[[1]]
          b   <- layer[[2]]
          out <- tf$add(tf$matmul(out, W), b)
          out <- self$activation(out)
        }
        
        out
      },
      
      compute_output_shape = function(input_shape) {
        
        output_dim <- self$hidden_dims[[length(hidden_dims)]]
        
        list(input_shape[[1]], output_dim)
      }
      
    )
  )

layer_encoder_decoder <-
  function(object,
           mode = 'encoder',
           num_layers = NULL,
           hidden_dims = NULL,
           original_dim = NULL,
           code_dim = NULL,
           activation = 'relu',
           name = NULL,
           trainable = TRUE) {
    
    create_layer(EncoderDecoderV2,
                 object,
                 list(
                   mode = tolower(mode),
                   num_layers = as.integer(num_layers),
                   hidden_dims = as.integer(hidden_dims),
                   original_dim = as.integer(original_dim),
                   code_dim = as.integer(code_dim),
                   activation = tf$keras$activations$get(activation),
                   name = name,
                   trainable = trainable
                 ))
  }


# TODO: Keras model that includes a matching encoder and decoder component
ae <- function() {
  input <- layer_input(shape = list(784))
  
  original_dim <- input$shape[[1]]
  hidden_dims  <- c(512, 256, 64)
  
  enc_out <- input %>% 
    layer_encoder_decoder(mode = "encoder",
                          num_layers = 3,
                          hidden_dims = hidden_dims)
  
  hidden_dims <- rev(hidden_dims)
  code_dim    <- enc_out$shape[[1]]
    
  dec_out <- enc_out %>% 
    layer_encoder_decoder(mode = "decoder",
                          num_layers = 3,
                          hidden_dims = hidden_dims,
                          original_dim = original_dim,
                          code_dim = code_dim)
  
  build_and_compile(input, dec_out)
}



# Model version of above
autoencoder_model <- 
  function(num_layers,
           hidden_dim,
           original_dim, # need to know at model instantiation
           name = NULL) {
    
    is_lst <- is_vec2(hidden_dim)
    
    hidden_dim <- 
      if(!is_lst) as.integer(hidden_dim) else hidden_dim
    
    num_layers <- as.integer(num_layers)
    original_dim <- as.integer(original_dim)
    
    stopifnot(is_lst & length(hidden_dim) == num_layers)
    
    keras_model_custom(name = name, function(self) {
      
      map2_fn <- function(mode, hidden) 
        layer_encoder_decoder(mode = mode, hidden_dim = hidden)
      
      self$encoder_layers <- purrr::map2(
        .x = rep('encoder', num_layers),
        .y = if (is_lst)
          hidden_dim
        else
          rep(hidden_dim, num_layers),
        .f = map2_fn
      )
      
      self$decoder_layers <- purrr::map2(
        .x = rep('decoder', num_layers + 1L),
        .y = if (is_lst)
          c(rev(hidden_dim), original_dim)
        else
          rep(hidden_dim, num_layers + 1L),
        .f = map2_fn
      )
      
      # Call
      function(x, mask = NULL) {
        
        output <- x
        
        # TODO: Shape issue??
        #'  Error in py_call_impl(callable, dots$args, dots$keywords) : 
        #InvalidArgumentError: 2 root error(s) found.
        #(0) Invalid argument: Incompatible shapes: [32,784] vs. [32,64]
        #[[{{node loss_2/output_1_loss/SquaredDifference}}]]
        #[[metrics_4/acc/Identity/_181]]
        #(1) Invalid argument: Incompatible shapes: [32,784] vs. [32,64]
        #[[{{node loss_2/output_1_loss/SquaredDifference}}]]
        #0 successful operations.
        #0 derived errors ignored. 
        
        for (i in 1L:length(self$encoder_layers) - 1L) { 
          output <- self$encoder_layers[i](output) 
        }
        
        for (j in 1L:length(self$decoder_layers) - 1L) { 
          output <- self$decoder_layers[i](output) 
        }
        
        output
      }
      
    })
  }

model <- 
  autoencoder_model(
    num_layers = 3,
    hidden_dim = list(512L, 256L, 64L),
    original_dim = 784
  )

model %>% compile(
  loss = 'mse',
  optimizer = 'adagrad',
  metrics = c('accuracy')
)



model %>% fit(x_train, 
              x_train, 
              epochs = 10, 
              batch_size = 128, 
              validation_data = list(x_test, x_test))






############################################################################
############################################################################



# 
# 
# Autoencoder_keras <-
#   R6::R6Class(
#     "Autoencoder_keras",
#     
#     inherit = KerasLayer,
#     
#     public = list(
#       hidden_dim = NULL,
#       encoder = NULL,
#       decoder = NULL,
#       original_dim = NULL,
#       
#       initialize = function(hidden_dim) {
#         self$hidden_dim <- hidden_dim
#       },
#       
#       build = function(input_shape) {
#         self$original_dim <- input_shape[[length(input_shape)]]
#         
#         self$encoder <-
#           layer_encoder_decoder(mode = 'encoder',
#                                 hidden_dim = self$hidden_dim)
#         
#         self$decoder <-
#           layer_encoder_decoder(
#             mode = 'decoder',
#             hidden_dim = self$hidden_dim,
#             original_dim = self$original_dim
#           )
#       },
#       
#       call = function(x, mask = NULL) {
#         encoder_out <- self$encoder(x)
#         decoder_out <- self$decoder(encoder_out)
#         
#         decoder_out
#       },
#       
#       compute_output_shape = function(input_shape) {
#         input_shape
#       }
#       
#     )
#   )


# layer_autoencoder <-
#   function(object,
#            hidden_dim,
#            name = NULL,
#            trainable = TRUE) {
#     create_layer(Autoencoder_keras,
#                  object,
#                  list(
#                    hidden_dim = as.integer(hidden_dim),
#                    name = name,
#                    trainable = trainable)
#     )
#   }

# la <- layer_autoencoder()
# la$build(y$shape)
# la$call(y)


# make_autoencoder <- 
#   function(input_dim = list(8192L)) {
#     input <- layer_input(shape = input_dim)
#     
#     output <- input %>% 
#       layer_autoencoder()
#     
#     build_and_compile(input, output, metric = 'mse')
#   }
# 
# make_autoencoder()
