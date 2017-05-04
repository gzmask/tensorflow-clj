(ns tensorflow-clj.core
  (:import [org.tensorflow TensorFlow Tensor Session Output Operation OperationBuilder Graph DataType])
  (:require [tensorflow-clj.helpers :as tf]
            [tensorflow-clj.utils :as utils :refer [tensor->clj clj->tensor]])
  (:gen-class))

(. TensorFlow version)

;; (def graph (new Graph))

;; (def tensor-1
;;   (let [tensor
;;         (Tensor/create
;;          (int-array
;;           [360 909 216 108 777 132 256 174 999 228 324 800 264]))]
;;     (-> graph
;;         (.opBuilder "Const" "tensor-1")
;;         (.setAttr "dtype" (.dataType tensor))
;;         (.setAttr "value" tensor)
;;         .build
;;         (.output 0))))

;; (def tensor-2
;;   (let [tensor
;;         (Tensor/create
;;          (int-array [5 9 2 1 7 3 8 2 9 2 3 8 8]))]
;;     (-> graph
;;         (.opBuilder "Const" "tensor-2")
;;         (.setAttr "dtype" (.dataType tensor))
;;         (.setAttr "value" tensor)
;;         .build
;;         (.output 0))))

;; (def divide
;;   (->
;;    (.opBuilder graph "Div" "my-dividing-operation")
;;    (.addInput tensor-1)
;;    (.addInput tensor-2)
;;    .build
;;    (.output 0)))

;; (def session (new Session graph))

;; (def result
;;   (-> session
;;       .runner
;;       (.fetch "my-dividing-operation")
;;       .run
;;       (.get 0)
;;       (.copyTo (int-array 13))))

;; (apply str (map char result))

(def training-data
  ;; input => output
  [[0. 0. 1.]   [0.]
   [0. 1. 1.]   [1.]
   [1. 1. 1.]   [1.]
   [1. 0. 1.]   [0.]])

(def inputs (tf/constant (take-nth 2 training-data)))
(def outputs (tf/constant (take-nth 2 (rest training-data))))

;; initial weights for the network
(def weights
  (tf/variable
   (repeatedly 3 (fn [] (repeatedly 1 #(dec (rand 2)))))))

;; forward propagation for the network to get the hypothesises of inputs
(defn network [x]
  (tf/sigmoid (tf/matmul x weights)))

;; this should be the cost function
(defn error [network-output]
  (tf/div (tf/pow (tf/sub outputs network-output) (tf/constant 2.)) (tf/constant 2.0)))

;; for back propagation: different between hypothesises output and the labeled correct output
(defn error' [network-output]
  (tf/sub network-output outputs))

;; reverse sigmoid function
(defn sigmoid' [x]
  (tf/mult x (tf/sub (tf/constant 1.) x)))

;; back propagation into layers closer to the front/input layer, in this case just one layer
(defn deltas [network-output]
  (tf/matmul
   (tf/transpose inputs)
   (tf/mult
    (error' (network inputs))
    (sigmoid' (network inputs)))))

(def train-network
  (tf/assign weights (tf/sub weights (deltas (network inputs)))))

;; pre-train states
(tf/session-run
 [(tf/global-variables-initializer)
  (network inputs)])

(def sess (tf/session))

(def sess-run (partial tf/session-run tf/default-graph sess))

(sess-run [(tf/global-variables-initializer)])

;; Run the train-network operation 10000 times and then check the error.
(sess-run
 [(repeat 10000 train-network)
  (tf/mean (error (network inputs)))])

;; test the trained network:
(sess-run [(network (tf/constant [[0. 0. 1.]]))])
(sess-run [(network (tf/constant [[0. 1. 1.]]))])
(sess-run [(network (tf/constant [[1. 1. 1.]]))])
(sess-run [(network (tf/constant [[1. 0. 1.]]))])
;; expected result:
;; [[0. 0. 1.]   [0.]
;;  [0. 1. 1.]   [1.]
;;  [1. 1. 1.]   [1.]
;;  [1. 0. 1.]   [0.]]

;; all in once:
(sess-run [(network inputs)])


(defn -main
  [& args]
  (println (sess-run [(network inputs)]))
  )
