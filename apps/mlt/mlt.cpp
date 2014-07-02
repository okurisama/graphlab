#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

#include <stdexcept>
#include <sstream>

#include "ublas_vector_serialization.hpp"

#include <graphlab.hpp>

using namespace boost::numeric;

// Global random reset probability
double RESET_PROB = 0.15;

double TOLERANCE = 1.0E-2;

size_t ITERATIONS = 0;

size_t TOP_K = 0;

//bool USE_DELTA_CACHE = false;

typedef std::vector<graphlab::vertex_id_type> id_vector_type;
id_vector_type SEEDS;

// The vertex data is a personalized-pagerank value per specified vertex
typedef float storage_scalar_type; // use float to save space in storage
typedef ublas::vector<storage_scalar_type> storage_vector_type;
const double REL_TOLERANCE = 1e-6; // appx rel precision of float

typedef double gather_scalar_type; // use double to preserve precision during summation
typedef ublas::vector<gather_scalar_type> gather_vector_type;

gather_vector_type N_VERTICES;


// The graph type is determined by the vertex and edge data types
typedef graphlab::distributed_graph<storage_vector_type, storage_scalar_type> graph_type;

void init_uniform_weights(graph_type::edge_type& edge) {
    edge.data() = 1.0/edge.source().num_out_edges();
}

void length_norm(graph_type::edge_type& edge) {
    // cf. T11106.  Assumes edge weights are normalized to be in [0,1] before application
    const gather_scalar_type n = edge.target().num_in_edges();
    edge.data() *= 10. / (1. + std::exp(4.-n)) / std::sqrt(std::max(100., n));
}

void one_vertex(graph_type::vertex_type& vertex) {
    vertex.data() = ublas::scalar_vector<storage_scalar_type>(std::max(1ul, SEEDS.size()));
}
void zero_vertex(graph_type::vertex_type& vertex) {
    vertex.data() = ublas::zero_vector<storage_scalar_type>(std::max(1ul, SEEDS.size()));
}

class normalize_edges : public graphlab::ivertex_program<graph_type, gather_scalar_type>
{
    gather_scalar_type total_;
public:
    normalize_edges() : total_(0) {}
    edge_dir_type gather_edges(icontext_type& context,
            const vertex_type& vertex) const {
        return graphlab::OUT_EDGES;
    }
    /* Add up the weight on out edges */
    gather_scalar_type gather(icontext_type& context, const vertex_type& vertex,
            edge_type& edge) const {
        return edge.data();
    }
    /* record the sum */
    void apply(icontext_type& context, vertex_type& vertex, const gather_scalar_type& total) {
        total_ = total;
    }
    edge_dir_type scatter_edges(icontext_type& context, const vertex_type& vertex) const {
        return total_ ? graphlab::OUT_EDGES : graphlab::NO_EDGES;
    }
    /* Normalize edge weights */
    void scatter(icontext_type& context, const vertex_type& vertex, edge_type& edge) const {
        edge.data() /= total_;
    }
    void save(graphlab::oarchive& oarc) const {
        oarc << total_;
    }
    void load(graphlab::iarchive& iarc) {
        iarc >> total_;
    }
};

// initially 1 for vertices that are reachable from a seed; 0 otherwise
class init_pageranks : public graphlab::ivertex_program<graph_type, storage_vector_type>
{
    bool changed;
public:
    init_pageranks() : changed(false) {}
    edge_dir_type gather_edges(icontext_type& context, const vertex_type& vertex) const {
        return graphlab::IN_EDGES;
    }
    storage_vector_type gather(icontext_type& context, const vertex_type& vertex, edge_type& edge) const {
        return edge.source().data();
    }
    void apply(icontext_type& context, vertex_type& vertex, const storage_vector_type& total) {
        storage_vector_type& before = vertex.data();
        storage_vector_type after = vertex.data();
        for (size_t i=0; i<total.size(); ++i) {
            after[i] = total[i] ? 1. : after[i];
        }
        const typename id_vector_type::iterator pos = std::lower_bound(SEEDS.begin(), SEEDS.end(), vertex.id());
        if (pos != SEEDS.end() && *pos == vertex.id()) {
            after[std::distance(SEEDS.begin(), pos)] = 1;
        }
        changed = false;
        for (size_t i=0; i<before.size(); ++i) {
            if (before[i] != after[i]) {
                changed = true;
                std::swap(before, after);
                break;
            }
        }
    }
    edge_dir_type scatter_edges(icontext_type& context, const vertex_type& vertex) const {
        return changed ? graphlab::OUT_EDGES : graphlab::NO_EDGES;
    }
    void scatter(icontext_type& context, const vertex_type& vertex, edge_type& edge) const {
        context.signal(edge.target());
    }
    void save(graphlab::oarchive& oarc) const {
        oarc << changed;
    }
    void load(graphlab::iarchive& iarc) {
        iarc >> changed;
    }
};

// MLT is really just weighted personalized pagerank
class mlt : public graphlab::ivertex_program<graph_type, gather_vector_type>
{
    bool converged;
public:
    mlt() : converged(false) {}
    edge_dir_type gather_edges(icontext_type& context,
            const vertex_type& vertex) const {
        return graphlab::IN_EDGES;
    }
    gather_vector_type gather(icontext_type& context, const vertex_type& vertex, edge_type& edge) const {
        return edge.source().data() * edge.data();
    }
    void apply(icontext_type& context, vertex_type& vertex, const gather_vector_type& total) {
        storage_vector_type& oldval = vertex.data();
        storage_vector_type newval = total.empty()
                                     ? storage_vector_type(ublas::zero_vector<storage_scalar_type>(std::max(1ul, SEEDS.size())))
                                     : total * (1.0 - RESET_PROB);
        if (SEEDS.empty()) {
            // non-personalized; so everyone gets some rank from teleportation
            newval[0] += RESET_PROB;
        } else {
            // seeds get teleportation rank from every vertex
            // non-seeds get nothing
            const typename id_vector_type::iterator pos = std::lower_bound(SEEDS.begin(), SEEDS.end(), vertex.id());
            if (pos != SEEDS.end() && *pos == vertex.id()) {
                const size_t i = std::distance(SEEDS.begin(), pos);
                newval[i] += RESET_PROB*N_VERTICES[i];
            }
        }
        converged = true;
        for (size_t i=0; i<newval.size(); ++i) {
            double delta = std::fabs(newval[i]-oldval[i]);
            if (delta > TOLERANCE && delta/newval[i] > REL_TOLERANCE) {
                converged = false;
                break;
            }
        }
        std::swap(oldval, newval);
        if (ITERATIONS) context.signal(vertex);
    }
    edge_dir_type scatter_edges(icontext_type& context, const vertex_type& vertex) const {
        if (ITERATIONS) return graphlab::NO_EDGES;
        return converged ? graphlab::NO_EDGES : graphlab::OUT_EDGES;
    }
    void scatter(icontext_type& context, const vertex_type& vertex, edge_type& edge) const {
        context.signal(edge.target());
    }
    void save(graphlab::oarchive& oarc) const {
        if (ITERATIONS == 0) oarc << converged;
    }
    void load(graphlab::iarchive& iarc) {
        if (ITERATIONS == 0) iarc >> converged;
    }
};


/*
 * We want to save the final graph so we define a write which will be
 * used in graph.save("path/prefix", mlt_writer()) to save the graph.
 */
struct mlt_writer
{
    std::string save_vertex(graph_type::vertex_type v) {
        std::stringstream strm;
        strm << v.id();
        for (size_t i=0; i<v.data().size(); ++i) {
            strm << " " << v.data()[i];
        }
        strm << std::endl;
        return strm.str();
    }
    std::string save_edge(graph_type::edge_type e) { return ""; }
}; // end of pagerank writer


bool wtsv_parser(graph_type& graph, const std::string& srcfilename, const std::string& str) {
    if (str.empty()) return true;
    size_t source=-1, target=-1;
    gather_scalar_type weight=0;
    char* targetptr;
    source = strtoul(str.c_str(), &targetptr, 10);
    if (targetptr == NULL) return false;
    target = strtoul(targetptr, &targetptr, 10);
    if (targetptr == NULL) return false;
    weight = strtod(targetptr, NULL);
    if(source != target && weight) graph.add_edge(source, target, weight);
    return true;
}

typedef std::pair<graphlab::vertex_id_type, storage_scalar_type> heap_entry_type;
typedef std::vector<heap_entry_type> heap_type;

struct MergeableHeaps : std::pair<std::vector<heap_type>,std::vector<heap_type> > // first deviations; second collections
{
private:
    static bool cmp(const heap_entry_type& a, const heap_entry_type& b) {
        return a.second>b.second || (a.second==b.second && a.first>b.first);
    }
public:
    void operator+=(const MergeableHeaps& other_pair) {
        for (size_t heapidx=0; heapidx<2; ++heapidx) {
            std::vector<heap_type>& me = heapidx ? second : first;
            const std::vector<heap_type>& other = heapidx ? other_pair.second : other_pair.first;
            for (size_t i=0; i<other.size(); ++i) {
                heap_type& heap = me[i];
                const heap_type& h2 = other[i];
                heap.reserve(heap.size()+other.size());
                std::copy(h2.begin(), h2.end(), std::back_inserter(heap));
                if (heap.size() > TOP_K) {
                    std::nth_element(heap.begin(), heap.begin()+TOP_K, heap.end(), cmp);
                    heap.resize(TOP_K);
                }
                std::make_heap(heap.begin(), heap.end(), cmp);
            }
        }
    }
    void save(graphlab::oarchive& oarc) const {
        oarc << (const std::pair<std::vector<heap_type>,std::vector<heap_type> >&)(*this);
    }
    void load(graphlab::iarchive& iarc) {
        iarc >> (std::pair<std::vector<heap_type>,std::vector<heap_type> >&)(*this);
    }
    void sort() {
        for (size_t heapidx=0; heapidx<2; ++heapidx) {
            std::vector<heap_type>& me = heapidx ? second : first;
            for (size_t i=0; i<me.size(); ++i) {
                std::sort_heap(me[i].begin(), me[i].end(), cmp);
            }
        }
    }
    static void map(const graph_type::vertex_type& v, MergeableHeaps& heaps_pair) {
        std::vector<heap_type>& heaps = (v.id() >= (1u<<31)) ? heaps_pair.second : heaps_pair.first;
        const size_t size = std::max(1ul, SEEDS.size());
        if (heaps.size() != size) {
            heaps.resize(size);
            for (size_t i=0; i<size; ++i) {
                heaps[i].reserve(TOP_K+1);
            }
        }
        for (size_t i=0; i<size; ++i) {
            heap_type& heap = heaps[i];
            heap_entry_type entry(v.id(), v.data()[i]);
            if (heap.size() < TOP_K || cmp(entry, heap.front())) {
                heap.push_back(entry);
                std::push_heap(heap.begin(), heap.end(), cmp);
                if (heap.size() > TOP_K) {
                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.pop_back();
                }
            }
        }
    }
};

gather_vector_type pagerank_sum(const graph_type::vertex_type& v) { return v.data(); }
gather_scalar_type weight_sum(const graph_type::edge_type& e) { return (gather_scalar_type)e.data(); }


int main(int argc, char** argv) {
    // Initialize control plain using mpi
    graphlab::mpi_tools::init(argc, argv);
    graphlab::distributed_control dc;
    global_logger().set_log_level(LOG_INFO);

    bool lengthnorm = false;

    // Parse command line options -----------------------------------------------
    graphlab::command_line_options clopts("MLT algorithm.");
    std::string graph_dir;
    std::string format = "adj";
    std::string exec_type = "synchronous";
    clopts.attach_option("graph", graph_dir,
            "The graph file.  If none is provided "
            "then a toy graph will be created");
    clopts.add_positional("graph");
    clopts.attach_option("engine", exec_type,
            "The engine type synchronous or asynchronous");
    clopts.attach_option("tol", TOLERANCE,
            "The permissible change at convergence.");
    clopts.attach_option("rp", RESET_PROB,
            "The reset probability.");
    clopts.attach_option("topk", TOP_K,
            "The top-k ranks to print");
    clopts.attach_option("format", format,
            "The graph file format -- unused!");
    clopts.attach_option("seeds", SEEDS,
            "vertices to use for personalization; if empty, run unpersonalized");
    clopts.attach_option("ln", lengthnorm,
            "whether to compute length normalization for edge weights");
    size_t powerlaw = 0;
    clopts.attach_option("powerlaw", powerlaw,
            "Generate a synthetic powerlaw out-degree graph. ");
    clopts.attach_option("iterations", ITERATIONS,
            "If set, will force the use of the synchronous engine"
            "overriding any engine option set by the --engine parameter. "
            "Runs complete (non-dynamic) PageRank for a fixed "
            "number of iterations. Also overrides the iterations "
            "option in the engine");
    //  clopts.attach_option("use_delta", USE_DELTA_CACHE,
    //                       "Use the delta cache to reduce time in gather.");
    std::string saveprefix;
    clopts.attach_option("saveprefix", saveprefix,
            "If set, will save the resultant pagerank to a "
            "sequence of files with prefix saveprefix");

    if(!clopts.parse(argc, argv)) {
        dc.cout() << "Error in parsing command line arguments." << std::endl;
        return EXIT_FAILURE;
    }

    std::sort(SEEDS.begin(), SEEDS.end());

    // Enable gather caching in the engine
    // doesn't work in the toolkit!  never converges...
    //  clopts.get_engine_args().set_option("use_cache", USE_DELTA_CACHE);

    if (ITERATIONS) {
        // make sure this is the synchronous engine
        dc.cout() << "--iterations set. Forcing Synchronous engine, and running "
                << "for " << ITERATIONS << " iterations." << std::endl;
        clopts.get_engine_args().set_option("type", "synchronous");
        clopts.get_engine_args().set_option("max_iterations", ITERATIONS);
        clopts.get_engine_args().set_option("sched_allv", true);
    }

    // Build the graph ----------------------------------------------------------
    graph_type graph(dc, clopts);
    if(powerlaw > 0) { // make a synthetic graph
        dc.cout() << "Loading synthetic Powerlaw graph." << std::endl;
        graph.load_synthetic_powerlaw(powerlaw, false, 2.1, 100000000);
    }
    else if (graph_dir.length() > 0) { // Load the graph from a file
        //    dc.cout() << "Loading graph in format: "<< format << std::endl;
        dc.cout() << "Loading graph in format: wtsv" << std::endl;
        graph.load(graph_dir, wtsv_parser);
    }
    else {
        dc.cout() << "graph or powerlaw option must be specified" << std::endl;
        clopts.print_description();
        return 0;
    }
    // must call finalize before querying the graph
    graph.finalize();
    {
        gather_scalar_type edgewtsum = graph.map_reduce_edges<gather_scalar_type>(weight_sum);
        dc.cout() << "Initial edge weight sum = " << edgewtsum/graph.num_vertices() << " * " << graph.num_vertices() << std::endl;
    }

    // initialize edge data
    if(powerlaw > 0) {
        // Initialize edge weights to uniform if we're using a toy graph
        graph.transform_edges(init_uniform_weights);
    } else {
        // Normalize edge weights
        graphlab::omni_engine<normalize_edges> engine(dc, graph, exec_type, clopts);
        engine.signal_all();
        engine.start();
        const double runtime = engine.elapsed_seconds();
        dc.cout() << "Finished normalizing weights in " << runtime
                << " seconds." << std::endl;
    }

    // do length norm
    if (lengthnorm) {
        graph.transform_edges(length_norm);
        dc.cout() << "Finished length norm" << std::endl;
        // need to renormalize
        graphlab::omni_engine<normalize_edges> engine(dc, graph, exec_type, clopts);
        engine.signal_all();
        engine.start();
        const double runtime = engine.elapsed_seconds();
        dc.cout() << "Finished renormalizing weights in " << runtime
                << " seconds." << std::endl;
    }

    {
        gather_scalar_type edgewtsum = graph.map_reduce_edges<gather_scalar_type>(weight_sum);
        dc.cout() << "Final edge weight sum = " << edgewtsum/graph.num_vertices() << " * " << graph.num_vertices() << std::endl;
    }

    // initialize vertex data
    if (SEEDS.empty()) {
        graph.transform_vertices(one_vertex); // init to 1
    } else {
        graph.transform_vertices(zero_vertex); // init to 0
        // set to 1 for anything that's connected to a seed
        graphlab::omni_engine<init_pageranks> engine(dc, graph, exec_type, clopts);
        for (size_t i=0; i<SEEDS.size(); ++i) {
            engine.signal(SEEDS[i]);
        }
        engine.start();
        const double runtime = engine.elapsed_seconds();
        dc.cout() << "Finished initializing pageranks in " << runtime
                << " seconds." << std::endl;
    }
    N_VERTICES = graph.map_reduce_vertices<gather_vector_type>(pagerank_sum);
    dc.cout() << "Initial pagerank sum = [";
    for (size_t i=0; i<N_VERTICES.size(); ++i) {
        dc.cout() << " " << N_VERTICES[i]/graph.num_vertices();
    }
    dc.cout() << " ] * " << graph.num_vertices() << std::endl;

    // actual MLT! (personalized pagerank)
    {
        graphlab::omni_engine<mlt> engine(dc, graph, exec_type, clopts);
        engine.signal_all();
        engine.start();
        const double runtime = engine.elapsed_seconds();
        dc.cout() << "Finished Running engine in " << runtime
                << " seconds." << std::endl;
    }

    N_VERTICES = graph.map_reduce_vertices<gather_vector_type>(pagerank_sum);
    dc.cout() << "Final pagerank sum = [";
    for (size_t i=0; i<N_VERTICES.size(); ++i) {
        dc.cout() << " " << N_VERTICES[i]/graph.num_vertices();
    }
    dc.cout() << " ] * " << graph.num_vertices() << std::endl;

    // get the top-ranking vertices for each personal seed
    if (TOP_K) {
        MergeableHeaps heaps = graph.fold_vertices<MergeableHeaps>(MergeableHeaps::map);
        heaps.sort();
        dc.cout() << "Top Deviations" << std::endl;
        for (size_t i=0; i<heaps.first.size(); ++i) {
            dc.cout() << "Seed " << i << std::endl;
            for (size_t j=0; j<heaps.first[i].size(); ++j) {
                dc.cout() << "\t:thumb" << heaps.first[i][j].first << ":\t" << heaps.first[i][j].second << std::endl;
            }
        }
        dc.cout() << std::endl << "Top Collections" << std::endl;
        for (size_t i=0; i<heaps.second.size(); ++i) {
            dc.cout() << "Seed " << i << std::endl;
            for (size_t j=0; j<heaps.second[i].size(); ++j) {
                dc.cout() << "\tCollection " << (heaps.second[i][j].first & ((1u<<31)-1u)) << ":\t" << heaps.second[i][j].second << std::endl;
            }
        }
    }

    // Save the final graph -----------------------------------------------------
    if (saveprefix != "") {
        graph.save(saveprefix, mlt_writer(),
                false,    // do not gzip
                true,     // save vertices
                false);   // do not save edges
    }

    // Tear-down communication layer and quit -----------------------------------
    graphlab::mpi_tools::finalize();
    return EXIT_SUCCESS;
} // End of main
