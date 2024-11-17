
#ifndef MATHLIB_PRIVATE_H
#define MATHLIB_PRIVATE_H

#ifdef HAVE_LONG_DOUBLE
#define LDOUBLE long double
#else
#define LDOUBLE double
#endif

// to ensure atanpi, cospi,  sinpi, tanpi are defined
#ifndef __STDC_WANT_IEC_60559_FUNCS_EXT__
#define __STDC_WANT_IEC_60559_FUNCS_EXT__ 1
#endif

#include <math.h>
#include <float.h>

#ifndef M_E
#define M_E 2.718281828459045235360287471353 /* e */
#endif

#ifndef M_LOG2E
#define M_LOG2E 1.442695040888963407359924681002 /* log2(e) */
#endif

#ifndef M_LOG10E
#define M_LOG10E 0.434294481903251827651128918917 /* log10(e) */
#endif

#ifndef M_LN2
#define M_LN2 0.693147180559945309417232121458 /* ln(2) */
#endif

#ifndef M_LN10
#define M_LN10 2.302585092994045684017991454684 /* ln(10) */
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383280 /* pi */
#endif

#ifndef M_2PI
#define M_2PI 6.283185307179586476925286766559 /* 2*pi */
#endif

#ifndef M_PI_2
#define M_PI_2 1.570796326794896619231321691640 /* pi/2 */
#endif

#ifndef M_PI_4
#define M_PI_4 0.785398163397448309615660845820 /* pi/4 */
#endif

#ifndef M_1_PI
#define M_1_PI 0.318309886183790671537767526745 /* 1/pi */
#endif

#ifndef M_2_PI
#define M_2_PI 0.636619772367581343075535053490 /* 2/pi */
#endif

#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.128379167095512573896158903122 /* 2/sqrt(pi) */
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.414213562373095048801688724210 /* sqrt(2) */
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524400844362105 /* 1/sqrt(2) */
#endif

// R specific constants

#ifndef M_SQRT_3
#define M_SQRT_3 1.732050807568877293527446341506 /* sqrt(3) */
#endif

#ifndef M_SQRT_32
#define M_SQRT_32 5.656854249492380195206754896838 /* sqrt(32) */
#endif

#ifndef M_LOG10_2
#define M_LOG10_2 0.301029995663981195213738894724 /* log10(2) */
#endif

#ifndef M_SQRT_PI
#define M_SQRT_PI 1.772453850905516027298167483341 /* sqrt(pi) */
#endif

#ifndef M_1_SQRT_2PI
#define M_1_SQRT_2PI 0.398942280401432677939946059934 /* 1/sqrt(2pi) */
#endif

#ifndef M_SQRT_2dPI
#define M_SQRT_2dPI 0.797884560802865355879892119869 /* sqrt(2/pi) */
#endif

#ifndef M_LN_2PI
#define M_LN_2PI 1.837877066409345483560659472811 /* log(2*pi) */
#endif

#ifndef M_LN_SQRT_PI
#define M_LN_SQRT_PI 0.572364942924700087071713675677
#endif

#ifndef M_LN_SQRT_2PI
#define M_LN_SQRT_2PI 0.918938533204672741780329736406
#endif

#ifndef M_LN_SQRT_PId2
#define M_LN_SQRT_PId2 0.225791352644727432363097614947
#endif

#ifdef HAVE_NEARYINT
#define R_forceint(x) nearbyint()
#else
#define R_forceint(x) round(x)
#endif

#define R_nonint(x) (fabs((x) - R_forceint(x)) > 1e-7 * fmax2(1., fabs(x)))

#undef FALSE
#undef TRUE

typedef enum { FALSE = 0, TRUE } bool_t;

#include <stdio.h>
#include <stdlib.h> /* for exit */

#define MATHLIB_ERROR(fmt, x) { printf(fmt, x); exit(1); }
#define MATHLIB_WARNING(fmt, x) printf(fmt, x)
#define MATHLIB_WARNING2(fmt, x, x2) printf(fmt, x, x2)
#define MATHLIB_WARNING3(fmt, x, x2, x3) printf(fmt, x, x2, x3)
#define MATHLIB_WARNING4(fmt, x, x2, x3, x4) printf(fmt, x, x2, x3, x4)
#define MATHLIB_WARNING5(fmt, x, x2, x3, x4, x5) printf(fmt, x, x2, x3, x4, x5)

#define ISNAN(x) (isnan(x) != 0)

// isfinite is defined in <math.h> according to c99
#define R_FINITE(x) isfinite(x)

#define ML_POSINF (1.0 / 0.0)
#define ML_NEGINF ((-1.0) / 0.0)
#define ML_NAN (0.0 / 0.0)

#define _(String) String

#define ML_VALID(x) (!ISNAN(x))

#define ME_NONE 0       /* no error */
#define ME_DOMAIN 1     /* argument out of domain */
#define ME_RANGE 2      /* value out of range */
#define ME_NOCONV 4     /* process did not converge */
#define ME_PRECISION 8  /* does not have "full" precision */
#define ME_UNDERFLOW 16 /* and underflow occured (important for IEEE)*/

#define ML_ERR_return_NAN        \
    {                            \
        ML_ERROR(ME_DOMAIN, ""); \
        return ML_NAN;           \
    }

#define ML_ERROR(x, s)                                                          \
    {                                                                           \
        if (x > ME_DOMAIN)                                                      \
        {                                                                       \
            char *msg = "";                                                     \
            switch (x)                                                          \
            {                                                                   \
            case ME_DOMAIN:                                                     \
                msg = _("argument out of domain in '%s'\n");                    \
                break;                                                          \
            case ME_RANGE:                                                      \
                msg = _("value out of range in '%s'\n");                        \
                break;                                                          \
            case ME_NOCONV:                                                     \
                msg = _("convergence failed in '%s'\n");                        \
                break;                                                          \
            case ME_PRECISION:                                                  \
                msg = _("full precision may not have been achieved in '%s'\n"); \
                break;                                                          \
            case ME_UNDERFLOW:                                                  \
                msg = _("underflow occurred in '%s'\n");                        \
                break;                                                          \
            }                                                                   \
            MATHLIB_WARNING(msg, s);                                            \
        }                                                                       \
    }

/* wilcoxon rank sum distribution */
#define WILCOX_MAX 50

#ifdef HAVE_VISIBILITY_ATTRIBUTE
#define attribute_hidden __attribute__((visibility("hidden")))
#else
#define attribute_hidden
#endif

// chebyshev series

int attribute_hidden chebyshev_init(double *, int, double);
double attribute_hidden chebyshev_eval(double, const double *, const int);

// gamma and related functions

double attribute_hidden lgammacor(double); // log(gamma) correction
double attribute_hidden stirlerr(double);  // stirling expansion "error"
double attribute_hidden bd0(double, double);
attribute_hidden double d1mach(int i);
attribute_hidden int i1mach(int i);

// acs toms

void attribute_hidden
bratio(double a, double b, double x, double y, double *w, double *w1,
       int *ierr, int log_p);

#endif

// utilities for density functions (d), cumulative possibility functions (p),
// and left quartile function (q). modified from rmath's dpq.h

#define give_log log_p

#define R_D__0 (log_p ? ML_NEGINF : 0.)
#define R_D__1 (log_p ? 0. : 1.)
#define R_DT_0 (lower_tail ? R_D__0 : R_D__1)
#define R_DT_1 (lower_tail ? R_D__1 : R_D__0)
#define R_D_half (log_p ? -M_LN2 : 0.5)

/* Use 0.5 - p + 0.5 to perhaps gain 1 bit of accuracy */
#define R_D_Lval(p) (lower_tail ? (p) : (0.5 - (p) + 0.5)) /*  p  */
#define R_D_Cval(p) (lower_tail ? (0.5 - (p) + 0.5) : (p)) /*  1 - p */

#define R_D_val(x) (log_p ? log(x) : (x))                     /*  x  in pF(x,..) */
#define R_D_qIv(p) (log_p ? exp(p) : (p))                     /*  p  in qF(p,..) */
#define R_D_exp(x) (log_p ? (x) : exp(x))                     /*  exp(x) */
#define R_D_log(p) (log_p ? (p) : log(p))                     /*  log(p) */
#define R_D_Clog(p) (log_p ? log1p(-(p)) : (0.5 - (p) + 0.5)) /*  [log](1-p) */

// log(1 - exp(x))  in more stable form than log1p(- R_D_qIv(x)) :
#define R_Log1_Exp(x) ((x) > -M_LN2 ? log(-expm1(x)) : log1p(-exp(x)))

// log(1-exp(x)):  R_D_LExp(x) == (log1p(- R_D_qIv(x))) but even more stable:
#define R_D_LExp(x) (log_p ? R_Log1_Exp(x) : log1p(-x))

#define R_DT_val(x) (lower_tail ? R_D_val(x) : R_D_Clog(x))
#define R_DT_Cval(x) (lower_tail ? R_D_Clog(x) : R_D_val(x))
#define R_DT_qIv(p) (log_p ? (lower_tail ? exp(p) : -expm1(p)) : R_D_Lval(p))

#define R_DT_CIv(p) (log_p ? (lower_tail ? -expm1(p) : exp(p)) : R_D_Cval(p))

#define R_DT_exp(x) R_D_exp(R_D_Lval(x))  /* exp(x) */
#define R_DT_Cexp(x) R_D_exp(R_D_Cval(x)) /* exp(1 - x) */

#define R_DT_log(p) (lower_tail ? R_D_log(p) : R_D_LExp(p))  /*  log(p) in qF   */
#define R_DT_Clog(p) (lower_tail ? R_D_LExp(p) : R_D_log(p)) /*  log(1-p) in qF */
#define R_DT_Log(p) (lower_tail ? (p) : R_Log1_Exp(p))

#define R_Q_P01_check(p)              \
    if ((log_p && p > 0) ||           \
        (!log_p && (p < 0 || p > 1))) \
    ML_ERR_return_NAN

// Do the boundaries exactly for q*() functions :
// Often  _LEFT_ = ML_NEGINF , and very often _RIGHT_ = ML_POSINF;
//
// R_Q_P01_boundaries(p, _LEFT_, _RIGHT_)  : <==>
//
//     R_Q_P01_check(p);
//     if (p == R_DT_0) return _LEFT_ ;
//     if (p == R_DT_1) return _RIGHT_;
//
// the following implementation should be more efficient (less tests):

#define R_Q_P01_boundaries(p, _LEFT_, _RIGHT_)    \
    if (log_p)                                    \
    {                                             \
        if (p > 0)                                \
            ML_ERR_return_NAN;                    \
        if (p == 0) /* upper bound */             \
            return lower_tail ? _RIGHT_ : _LEFT_; \
        if (p == ML_NEGINF)                       \
            return lower_tail ? _LEFT_ : _RIGHT_; \
    }                                             \
    else                                          \
    { /* !log_p */                                \
        if (p < 0 || p > 1)                       \
            ML_ERR_return_NAN;                    \
        if (p == 0)                               \
            return lower_tail ? _LEFT_ : _RIGHT_; \
        if (p == 1)                               \
            return lower_tail ? _RIGHT_ : _LEFT_; \
    }

#define R_P_bounds_01(x, x_min, x_max) \
    if (x <= x_min)                    \
        return R_DT_0;                 \
    if (x >= x_max)                    \
    return R_DT_1

// is typically not quite optimal for (-Inf,Inf) where you'd rather have
#define R_P_bounds_Inf_01(x) \
    if (!R_FINITE(x))        \
    {                        \
        if (x > 0)           \
            return R_DT_1;   \
        return R_DT_0;       \
    }

// additions for density functions (C.Loader)
#define R_D_fexp(f, x) (give_log ? -0.5 * log(f) + (x) : exp(x) / sqrt(f))

// [neg]ative or [non int]eger :
#define R_D_negInonint(x) (x < 0. || R_nonint(x))

// for discrete d<distr>(x, ...) :
#define R_D_nonint_check(x)                          \
    if (R_nonint(x))                                 \
    {                                                \
        MATHLIB_WARNING(_("non-integer x = %f"), x); \
        return R_D__0;                               \
    }

// probability functions

double df(double x, double m, double n, int give_log);
double dgamma(double x, double shape, double scale, int give_log);
double dpois_raw(double x, double lambda, int give_log);
double dpois(double x, double lambda, int give_log);
double lgammafn_sign(double x, int *sgn);
double lgammafn(double x);
double dbinom_raw(double x, double n, double p, double q, int give_log);
double dbinom(double x, double n, double p, int give_log);
double dchisq(double x, double df, int give_log);
double pf(double x, double df1, double df2, int lower_tail, int log_p);
double pchisq(double x, double df, int lower_tail, int log_p);
static double logcf(double x, double i, double d, double eps);
double log1pmx(double x);
double lgamma1p(double a);
double logspace_add(double logx, double logy);
double logspace_sub(double logx, double logy);
double logspace_sum(const double *logx, int n);
static double dpois_wrap(double x_plus_1, double lambda, int give_log);
static double pgamma_smallx(double x, double alph, int lower_tail, int log_p);
static double pd_upper_series(double x, double y, int log_p);
static double pd_lower_cf(double y, double d);
static double pd_lower_series(double lambda, double y);
static double dpnorm(double x, int lower_tail, double lp);
static double ppois_asymp(double x, double lambda, int lower_tail, int log_p);
double pgamma_raw(double x, double alph, int lower_tail, int log_p);
double pgamma(double x, double alph, double scale, int lower_tail, int log_p);
double pbinom(double x, double n, double p, int lower_tail, int log_p);
attribute_hidden double pbeta_raw(double x, double a, double b, int lower_tail, int log_p);
double pbeta(double x, double a, double b, int lower_tail, int log_p);
double fmax2(double x, double y);
double sinpi(double x);
double tanpi(double x);
double cospi(double x);
double gammafn(double x);
double dnorm4(double x, double mu, double sigma, int give_log);
double pnorm5(double x, double mu, double sigma, int lower_tail, int log_p);
void pnorm_both(double x, double *cum, double *ccum, int i_tail, int log_p);
double dt(double x, double n, int give_log);
double pt(double x, double n, int lower_tail, int log_p);
double lbeta(double a, double b);

#define dnorm dnorm4
#define pnorm pnorm5