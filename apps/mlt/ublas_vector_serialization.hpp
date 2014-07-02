#ifndef GRAPHLAB_SERIALIZE_UBLAS_VECTOR_HPP
#define GRAPHLAB_SERIALIZE_UBLAS_VECTOR_HPP

#include <boost/numeric/ublas/vector.hpp>
#include <graphlab/serialization/iarchive.hpp>
#include <graphlab/serialization/oarchive.hpp>
#include <graphlab/serialization/iterator.hpp>

namespace graphlab {
  namespace archive_detail {
    /**
     * We re-dispatch vectors because based on the contained type,
     * it is actually possible to serialize them like a POD
     */
    template <typename OutArcType, typename ValueType, bool IsPOD>
    struct ublas_vector_serialize_impl {
      static void exec(OutArcType& oarc, const ValueType& vec) {
        // really this is an assert false. But the static assert
        // must depend on a template parameter
        BOOST_STATIC_ASSERT(sizeof(OutArcType) == 0);
        assert(false);
      };
    };
    /**
     * We re-dispatch vectors because based on the contained type,
     * it is actually possible to deserialize them like iarc POD
     */
    template <typename InArcType, typename ValueType, bool IsPOD>
    struct ublas_vector_deserialize_impl {
      static void exec(InArcType& iarc, ValueType& vec) {
        // really this is an assert false. But the static assert
        // must depend on a template parameter
        BOOST_STATIC_ASSERT(sizeof(InArcType) == 0);
        assert(false);
      };
    };

    /// Fast vector serialization if contained type is a POD
    template <typename OutArcType, typename ValueType>
    struct ublas_vector_serialize_impl<OutArcType, ValueType, true > {
      static void exec(OutArcType& oarc, const boost::numeric::ublas::vector<ValueType>& vec) {
        const size_t len = vec.size();
        oarc << len;
        if (len) {
          serialize(oarc, &(vec.data()[0]), sizeof(ValueType)*len);
        }
      }
    };

    /// Fast vector deserialization if contained type is a POD
    template <typename InArcType, typename ValueType>
    struct ublas_vector_deserialize_impl<InArcType, ValueType, true > {
      static void exec(InArcType& iarc, boost::numeric::ublas::vector<ValueType>& vec){
        size_t len;
        iarc >> len;
        vec.clear();
        vec.resize(len);
        if (len) {
          deserialize(iarc, &(vec.data()[0]), sizeof(ValueType)*len);
        }
      }
    };



    /**
       Serializes a vector */
    template <typename OutArcType, typename ValueType>
    struct serialize_impl<OutArcType, boost::numeric::ublas::vector<ValueType>, false > {
      static void exec(OutArcType& oarc, const boost::numeric::ublas::vector<ValueType>& vec) {
        ublas_vector_serialize_impl<OutArcType, ValueType,
          gl_is_pod_or_scaler<ValueType>::value >::exec(oarc, vec);
      }
    };
    /**
       deserializes a vector */
    template <typename InArcType, typename ValueType>
    struct deserialize_impl<InArcType, boost::numeric::ublas::vector<ValueType>, false > {
      static void exec(InArcType& iarc, boost::numeric::ublas::vector<ValueType>& vec){
        ublas_vector_deserialize_impl<InArcType, ValueType,
          gl_is_pod_or_scaler<ValueType>::value >::exec(iarc, vec);
      }
    };
  } // archive_detail
} // namespace graphlab

#endif

