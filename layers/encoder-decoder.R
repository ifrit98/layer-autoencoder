# TODO: Update encoder/decoder layers with batchnorm and dropout options

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


layer_encoder_decoder <-
  function(object,
           mode = 'encoder',
           hidden_dim = NULL,
           original_dim = NULL,
           name = NULL,
           trainable = TRUE) {
    
    create_layer(EncoderDecoder,
                 object,
                 list(
                   mode = tolower(mode),
                   hidden_dim = as.integer(hidden_dim),
                   original_dim = as.integer(original_dim),
                   name = name,
                   trainable = trainable
                 ))
  }



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
              initializer = initializer_he_normal(),
              trainable = TRUE)

            b <- self$add_weight(
              name = "b_",
              shape = list(out_features),
              initializer = initializer_zeros(),
              trainable = TRUE)

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

layer_encoder_decoderV2 <-
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




autoencoder_modelV2 <- 
  function(num_layers = 3,
           hidden_dims = c(512, 256, 64),
           original_dim = 784,
           name = NULL) {
    
    num_layers <- as.integer(num_layers)
    original_dim <- as.integer(original_dim)
    
    stopifnot(length(hidden_dims) == num_layers)
    
    keras_model_custom(name = name, function(self) {
      
      num_layers   <- as.integer(num_layers)
      original_dim <- as.integer(original_dim)
      hidden_dims  <- c(512, 256, 64)
      
      
      self$encoder_layer <- 
        layer_encoder_decoderV2(mode = "encoder",
                                num_layers = num_layers,
                                hidden_dims = hidden_dims)
      
      hidden_dims <- rev(hidden_dims)
      code_dim    <- hidden_dims[[1]]
      
      self$decoder_layer <- 
        layer_encoder_decoderV2(mode = "decoder",
                                num_layers = num_layers,
                                hidden_dims = hidden_dims,
                                original_dim = original_dim,
                                code_dim = code_dim)

      # Call
      function(x, mask = NULL) {
        browser()
        x <- self$encoder_layer(x)
        x <- self$decoder_layer(x)
        
        x
      }
    })
}


autoencoder_model <- 
  function(num_layers = 3,
           hidden_dim = c(512, 256, 64),
           original_dim = 784, # need to know at model instantiation
           name = NULL) {
    
    is_lst <- is_vec2(hidden_dim)
    
    hidden_dim <- 
      if(!is_lst) as.integer(hidden_dim) else hidden_dim
    
    num_layers   <- as.integer(num_layers)
    original_dim <- as.integer(original_dim)
    
    stopifnot(is_lst & length(hidden_dim) == num_layers)
    
    keras_model_custom(name = name, function(self) {
      
      map2_fn <- function(mode, hidden) {
        # browser() 
        layer_encoder_decoder(mode = mode, hidden_dim = hidden)
      }
    
      self$encoder_layers <- purrr::map2(
        .x = rep('encoder', num_layers),
        .y = if (is_lst)
          hidden_dim
        else
          rep(hidden_dim, num_layers),
        .f = map2_fn
      )
      
      # lst <- list(rev(hidden_dim), original_dim)
      # TODO: WHY THE FUCK IS DECODER MAP CALL REPLICATING NUMBERs 64 or 784?
      self$decoder_layers <- purrr::map2(
        .x = rep('decoder', num_layers + 1L),
        .y = c(rev(hidden_dim), original_dim), # if (is_lst) lst else rep(hidden_dim, num_layers + 1L),
        .f = map2_fn
      )
      # self$decoder_layers[[0]](x) %>% print()
      # self$decoder_layers[[1]](x) %>% print()
      # self$decoder_layers[[2]](x) %>% print()
      # self$decoder_layers[[3]](x) %>% print()
      # browser()
      
      # Call
      function(x, mask = NULL) {
        
        output <- x
        
        for (i in 1L:length(self$encoder_layers) - 1L) { 
          output <- self$encoder_layers[i](output) 
        }
        
        for (j in 1L:length(self$decoder_layers) - 1L) { 
          browser()
          output <- self$decoder_layers[j](output) 
        }
        
        output
      }
      
    })
  }

